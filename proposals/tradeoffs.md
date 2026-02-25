# Tradeoffs

## What You Gain

### 1. Composability
Users can build custom pipelines by mixing and matching transforms:

```csharp
// Standard embedding pipeline
var pipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolOpts));

// Same tokenizer + scorer, different pooling
var clsPipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(clsPoolOpts));

// Same tokenizer + scorer, but for classification (future)
var classifyPipeline = mlContext.Transforms.TokenizeText(tokOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
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

### 1. Memory: Intermediate IDataView Materialization

The monolith processes batches of 32 with method-local arrays. The modular ML.NET pipeline materializes ALL intermediate data as IDataView columns.

| Scenario | Monolith Peak Memory | Modular Peak Memory | Ratio |
|----------|---------------------|--------------------:|------:|
| 100 rows, unpooled model (384-dim, 128-seq) | ~6 MB | ~9 MB | 1.5x |
| 1K rows, unpooled model | ~6 MB | ~73 MB | 12x |
| 10K rows, unpooled model | ~6 MB | ~700 MB | 117x |
| 100K rows, unpooled model | ~6 MB | ~7 GB | 1167x |
| 10K rows, pre-pooled model | ~6 MB | ~20 MB | 3.3x |

**Calculation for unpooled model (10K rows):**
- TokenIds: 10K × 128 × 8 bytes = 10 MB
- AttentionMask: 10K × 128 × 8 bytes = 10 MB
- TokenTypeIds: 10K × 128 × 8 bytes = 10 MB
- RawOutput: 10K × 128 × 384 × 4 bytes = **1.9 GB** ← dominates
- But: these are sequential. TokenIds can be GC'd after scoring. Net peak ≈ 700 MB for tokens + raw output simultaneously.

**Mitigation:** The `GenerateEmbeddings()` direct face (used by MEAI generator and the facade) does NOT pay this cost. It uses the three transforms' internal direct faces which process batch-by-batch with transient arrays — same memory profile as the monolith.

**Who pays the cost:** Only users who use the composable ML.NET pipeline path (`mlContext.Transforms.TokenizeText().Append().Append()`). The convenience facade and MEAI path are unaffected.

### 2. Allocation Pressure

Each row gets its own per-row arrays for IDataView columns:
- `long[128]` for TokenIds (1 per row)
- `long[128]` for AttentionMask (1 per row)
- `long[128]` for TokenTypeIds (1 per row)
- `float[128 × 384]` for RawOutput (1 per row, unpooled)
- `float[384]` for Embedding (1 per row)

For 10K rows: ~50K array allocations (vs. ~300 per-batch allocations in the monolith).

**Mitigation:** Same as memory — direct faces avoid per-row allocation.

### 3. Batch Reconstruction Overhead

The monolith builds batch tensors directly from text. The modular path:
1. Tokenizer creates per-row arrays
2. Scorer reads per-row arrays from IDataView cursor
3. Scorer copies per-row arrays into batch-sized flat arrays for ONNX
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
var scorerOpts = new OnnxTextModelScorerOptions { TokenIdsColumnName = "InputIds" }; // ← mismatch!
```

**Mitigation:** Default column names are consistent across all three transforms (`"TokenIds"`, `"AttentionMask"`, `"TokenTypeIds"`, `"RawOutput"`, `"Embedding"`). Mismatches only occur if the user explicitly overrides names inconsistently. The facade eliminates this entirely by wiring up column names internally.

### 5. Complexity

6 new files (3 estimators, 3 transformers) + refactoring of 2 existing files, plus a new `TokenizedBatch` type and `OnnxModelMetadata` record.

**Mitigation:** Each file is focused and small. The individual transforms are simpler than the monolith because they each do one thing. Total code may increase by ~30%, but cyclomatic complexity per file decreases.

## Summary

| Dimension | Impact | Who's Affected | Mitigation |
|-----------|--------|---------------|------------|
| Memory (IDataView path) | 🔴 Significant for large datasets + unpooled models | Composable pipeline users only | Use facade/MEAI path; direct faces have same memory as monolith |
| Allocation pressure | 🟡 Moderate | Composable pipeline users only | Direct faces avoid per-row allocation |
| Batch copy overhead | 🟢 Negligible | Everyone using IDataView path | <0.1% of total processing time |
| Misconfiguration | 🟡 Moderate | Composable pipeline users | Consistent defaults; facade eliminates it |
| Complexity | 🟡 More files | Maintainers | Each file is simpler; single responsibility |
| Composability | 🟢 Major gain | All users | New capability |
| Reusability | 🟢 Major gain | Future task implementations | Tokenizer + scorer shared across tasks |
| Provider flexibility | 🟢 Major gain | Users wanting remote providers | EmbeddingGeneratorEstimator |
| Testability | 🟢 Moderate gain | Developers | Each transform testable in isolation |
