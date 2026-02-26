# Tradeoffs

## What You Gain

### 1. Composability
Users can build custom pipelines by mixing and matching transforms:

```csharp
// Standard embedding pipeline
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextEmbedding(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolOpts));

// Same tokenizer + scorer, different pooling
var clsPipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextEmbedding(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(clsPoolOpts));

// Same tokenizer + scorer, but for classification (future)
var classifyPipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextEmbedding(scorerOpts))
    .Append(mlContext.Transforms.SoftmaxClassify(classifyOpts));
```

### 2. Inspectability
Debug intermediate results by examining IDataView columns after each transform:

```csharp
var tokenized = tokenizerTransformer.Transform(data);
// Inspect: what tokens were produced? Is truncation happening?

var scored = scorerTransformer.Transform(tokenized);
// Inspect: what does the raw model output look like?

var embedded = poolingTransformer.Transform(scored);
// Inspect: final embedding values
```

### 3. Reusability
The tokenizer and scorer are universal for any transformer ONNX model task. Adding classification, NER, QA, or reranking only requires a new post-processing transform — the first two steps are shared.

### 4. Testability
Each transform has a narrow contract:
- Tokenizer: text in → tokens out. Test with known vocab + known text.
- Scorer: tokens in → tensor out. Test with known model + known tokens.
- Pooler: tensor in → embedding out. Test with synthetic tensors + known math.

### 5. Provider swappability (via EmbeddingGeneratorEstimator)
The provider-agnostic transform lets users swap embedding providers within ML.NET pipelines:

```csharp
// Local ONNX
var pipeline = mlContext.Transforms.TextEmbedding(onnxGenerator);

// Remote OpenAI (same pipeline structure, different provider)
var pipeline = mlContext.Transforms.TextEmbedding(openAIGenerator);
```

## What You Lose

### 1. ~~Memory: Intermediate IDataView Materialization~~ → SOLVED by Lazy Evaluation

The original concern was that eager modular transforms would materialize ALL intermediate data
as IDataView columns, leading to ~1.9 GB peak memory for 10K rows with unpooled models.

**Lazy evaluation eliminates this entirely.** Each transform's `Transform()` returns a wrapping
IDataView that computes values on-demand. The scorer cursor uses lookahead batching to maintain
ONNX batch throughput while only keeping one batch of data in memory.

| Scenario | Monolith | Modular Lazy | Ratio |
|----------|----------|-------------|------:|
| Any row count, unpooled model (384-dim, 128-seq) | ~6 MB | ~6 MB | 1.0x |
| Any row count, pre-pooled model | ~2 MB | ~2 MB | 1.0x |

Peak memory is bounded by `BatchSize × rowSize`, regardless of total row count. Same as the
current monolith.

### 2. Allocation Pressure — Bounded

With lazy evaluation, per-row arrays exist only for the current batch (32 rows by default).
Previous batches are eligible for GC. Total allocations per batch:

- Tokenizer: 32 × `long[128]` for TokenIds + AttentionMask = ~64 arrays
- Scorer: 32 × `float[seqLen × hiddenDim]` for raw output = ~32 arrays
- Pooler: 32 × `float[hiddenDim]` for embedding = ~32 arrays

This is ~128 arrays per batch cycle — comparable to the monolith's ~300 per-batch allocations.

### 3. Batch Reconstruction Overhead

The monolith builds batch tensors directly from text. The modular path:
1. Tokenizer cursor creates per-row arrays
2. Scorer cursor reads per-row arrays from upstream tokenizer cursor
3. Scorer cursor copies per-row arrays into batch-sized flat arrays for ONNX
4. After ONNX, scorer unpacks batch output into per-row arrays

Step 3 is an extra copy per direction that the monolith avoids.

**Impact:** Negligible compared to ONNX inference time. For a batch of 32:
- Copy overhead: ~0.01ms (memcpy of 32 × 128 × 8 bytes)
- ONNX inference: ~10-50ms

### 4. Misconfiguration Surface

Three options classes means three places where column names must agree:

```csharp
// If these don't match, you get a runtime error:
var tokOpts = new TextTokenizerOptions { TokenIdsColumnName = "TokenIds" };
var scorerOpts = new OnnxTextEmbeddingScorerOptions { TokenIdsColumnName = "InputIds" }; // ← mismatch!
```

**Mitigation:** Default column names are consistent across all three transforms (`"TokenIds"`, `"AttentionMask"`, `"TokenTypeIds"`, `"RawOutput"`, `"Embedding"`). Mismatches only occur if the user explicitly overrides names inconsistently. The facade eliminates this entirely by wiring up column names internally.

### 5. Implementation Complexity

Each transform requires custom `IDataView` + `DataViewRowCursor` + getter delegate boilerplate:

| Transform | Custom Types | Lines (est.) | Notes |
|-----------|-------------|-------------|-------|
| Tokenizer | `TokenizerDataView`, `TokenizerCursor` | ~150 | Simple row-by-row |
| Scorer | `ScorerDataView`, `ScorerCursor` | ~250 | Complex: lookahead batching + upstream caching |
| Pooler | `PoolerDataView`, `PoolerCursor` | ~120 | Simple row-by-row, passthrough delegation |

Total boilerplate: ~520 lines across the three transforms.

**This boilerplate is temporary scaffolding.** When migrating to ML.NET (Approach D), all
custom IDataView/cursor classes are deleted and replaced by `RowToRowTransformerBase` /
`MapperBase` overrides. See [migration-to-mlnet.md](migration-to-mlnet.md).

### 6. Scorer Cursor Complexity: Upstream Column Caching

The scorer cursor reads ahead N rows for lookahead batching. This means the upstream cursor
has advanced past those rows. When the downstream consumer (pooler) asks for passthrough
column values via the scorer cursor, we can't delegate to the upstream cursor — it's pointing
at a different row.

**Solution:** The scorer cursor caches all passthrough column values for the current batch.
This is ~32 rows of text strings and token arrays — typically < 1 MB. The caching logic adds
~50 lines of code and is the most subtle part of the implementation.

The pooler cursor does NOT have this problem because it processes in lockstep with upstream.

## Summary

| Dimension | Impact | Who's Affected | Mitigation |
|-----------|--------|---------------|------------|
| Memory | 🟢 Same as monolith (~6 MB) | Nobody | Lazy evaluation with lookahead batching |
| Allocation pressure | 🟢 Bounded per-batch | Nobody | Same order as monolith |
| Batch copy overhead | 🟢 Negligible | Everyone | <0.1% of total processing time |
| Misconfiguration | 🟡 Moderate | Composable pipeline users | Consistent defaults; facade eliminates it |
| Impl complexity | 🟡 ~520 lines boilerplate | Maintainers | Temporary — deleted in Approach D migration |
| Scorer caching | 🟡 Subtle logic | Maintainers | Well-isolated in ScorerCursor |
| Composability | 🟢 Major gain | All users | New capability |
| Reusability | 🟢 Major gain | Future task implementations | Tokenizer + scorer shared across tasks |
| Provider flexibility | 🟢 Major gain | Users wanting remote providers | EmbeddingGeneratorEstimator |
| Testability | 🟢 Moderate gain | Developers | Each transform testable in isolation |
