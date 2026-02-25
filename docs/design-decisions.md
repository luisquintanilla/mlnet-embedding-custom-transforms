# Design Decisions

This document explains *why* every major design choice was made. It's written for developers and AI coding agents who need to understand the trade-off space before modifying or extending the solution.

## The ML.NET Constraint Landscape

ML.NET's pipeline model is built on two interfaces: `IEstimator<TTransformer>` (learns from data, produces a transformer) and `ITransformer` (applies a transformation to an `IDataView`). When you need a transform that ML.NET doesn't provide — such as running text through a HuggingFace ONNX model — you must implement these interfaces yourself.

The challenge is that ML.NET's most convenient base classes are inaccessible from external code:

```
RowToRowTransformerBase          ← private protected constructor
OneToOneTransformerBase          ← private protected constructor
MapperBase / OneToOneMapperBase  ← private protected constructor
TrivialEstimator<T>              ← private protected constructor
```

These classes handle lazy evaluation (cursor-based streaming), schema propagation, and save/load via `[LoadableClass]` attributes. From an external project, you can't use any of them.

The [ML.NET Custom Transformer Guide](https://github.com/luisquintanilla/mlnet-custom-transformer-guide) documents four approaches:

| Approach | Pattern | External Project? | Lazy Eval | Save/Load | Limitation |
|----------|---------|:------------------:|:---------:|:---------:|------------|
| **A** | `CustomMapping` lambda | ✅ | ✅ | ⚠️ factory | Static POCO schema, no lifecycle hooks |
| **B** | Production Facade + `CustomMapping` | ✅ | ✅ | ⚠️ factory | Still POCO-static under the hood |
| **C** | Direct `IEstimator`/`ITransformer` | ✅ | ❌ eager | ❌ | Materializes all rows, no built-in save |
| **D** | `RowToRowTransformerBase` subclass | ❌ | ✅ | ✅ | Must be inside `dotnet/machinelearning` repo |

**None of the approaches give us everything we need.** Our requirements are:

1. ✅ External project (this is a prototype, not in the ML.NET repo)
2. ✅ Save/load (serialize to a portable model file)
3. ❌ No `CustomMapping` (static `[VectorType]` dimensions can't adapt to different models)
4. ✅ Resource management (ONNX InferenceSession, tokenizer lifecycle)

### Why We Chose Approach C Enhanced

Approach C gives us full control — we implement `IEstimator<T>` and `ITransformer` directly. The "enhanced" part is bolting on custom zip-based save/load since we can't use ML.NET's internal `ICanSaveModel` mechanism.

**Why not A/B (CustomMapping)?** The `CustomMapping` transform requires POCO classes with compile-time `[VectorType(N)]` attributes. For embedding models, `N` varies by model (384 for MiniLM, 768 for MPNet). You'd need to recompile for each model. Additionally, the `CustomMappingFactory` save/load pattern requires assembly scanning and static state for reconstruction — fragile and unintuitive.

**Why not D (internal base classes)?** This is a prototype. We want to iterate quickly without forking the ML.NET repo. If this proves valuable, the code can be ported to Approach D later (see [extending.md](extending.md)).

## Eager Evaluation with Configurable Batch Size

Approach C's default is to materialize all rows in `Transform()`. We chose this deliberately:

**Why eager?**
- ONNX inference is fundamentally batch-oriented — batching multiple texts into one `Run()` call is 5-10x faster than per-row inference
- For embedding generation, you typically process a known corpus (not an infinite stream)
- Implementing a custom `IDataView` with cursor-based lazy evaluation is ~300 lines of complex code (custom `DataViewSchema`, `RowCursor`, column getters with thread safety)

**The batch-size middle ground:**
Instead of loading ALL rows into memory at once, we process in configurable chunks (default: 32 rows). This bounds memory usage while maintaining batch throughput:

```csharp
for (int start = 0; start < texts.Count; start += batchSize)
{
    int count = Math.Min(batchSize, texts.Count - start);
    var batchEmbeddings = ProcessBatch(batchTexts);
    allEmbeddings.AddRange(batchEmbeddings);
}
```

**What we deferred:** Lazy cursor-based evaluation. A future implementation could wrap the input `IDataView` and compute embeddings on-demand as a cursor advances, potentially with lookahead batching. See [extending.md](extending.md) for the sketch.

## Save/Load Strategy

ML.NET's native `Model.Save()` calls `ICanSaveModel.Save(ModelSaveContext)` on each transformer in the chain. This interface is *internal* to ML.NET — external transformers cannot participate.

We evaluated three options:

| Option | Portable? | Size | `mlContext.Model.Save()` compatible? |
|--------|:---------:|:----:|:------------------------------------:|
| **A — Custom zip** | ✅ | ~80 MB | ❌ |
| B — Reference paths | ❌ | ~1 KB | ❌ |
| C — TransformerChain + CustomMapping | ✅ | ~80 MB | ✅ |

**We chose Option A** — a self-contained zip file containing:

```
embedding-model.mlnet (zip)
├── model.onnx        — The ONNX model file (copied verbatim)
├── vocab.txt         — The tokenizer vocabulary (original filename preserved)
├── config.json       — Serialized OnnxTextEmbeddingOptions
└── manifest.json     — Version info, embedding dimension, creation timestamp
```

**Why self-contained?** The ONNX model IS the model — it makes no sense to save a path reference that breaks when the file moves. The zip is ~80 MB for MiniLM (mostly the ONNX file), which is comparable to ML.NET's own saved models with embedded weights.

**Why not Option C?** It would require using `CustomMapping` internally (which the user explicitly ruled out) and the `CustomMappingFactory` assembly-scanning pattern for reconstruction.

**Loading:** `ModelPackager.Load()` extracts the zip to a temp directory, reads `config.json` to reconstruct `OnnxTextEmbeddingOptions`, then uses `OnnxTextEmbeddingEstimator.Fit()` to recreate the transformer with full auto-discovery.

## ONNX Auto-Discovery

Most ML.NET transforms require the user to manually specify input/output column mappings. We chose auto-discovery because sentence-transformer models follow a strong convention:

**Inputs:** `input_ids`, `attention_mask`, `token_type_ids` (optional)
**Outputs:** `last_hidden_state` (needs pooling) or `sentence_embedding` (pre-pooled)

The estimator probes `InferenceSession.InputMetadata` and `OutputMetadata` at `Fit()` time:

```csharp
// Discover inputs by convention
string inputIdsName = FindTensorName(inputMeta, ["input_ids"], "input_ids");
string attentionMaskName = FindTensorName(inputMeta, ["attention_mask"], "attention_mask");

// Discover outputs — prefer pre-pooled if available
var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
if (pooledName != null) { /* skip manual pooling */ }
else { /* use last_hidden_state + mean pooling */ }

// Embedding dimension from the last axis of the output tensor
int hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
```

This mirrors how ML.NET's own `OnnxTransformer` works internally — it creates an `OnnxModel` that inspects the ONNX graph for input/output metadata. The difference is that our estimator applies domain knowledge (sentence-transformer conventions) to provide zero-configuration defaults.

**Manual override:** Every auto-discovered value can be overridden via `OnnxTextEmbeddingOptions` for non-standard models.

## Thread Safety

**`InferenceSession`:** OnnxRuntime's documentation states that `Run()` is thread-safe for concurrent calls. The session handles internal locking. We use a single session per transformer instance.

**`Tokenizer`:** `Microsoft.ML.Tokenizers` tokenizers are stateless after construction. `EncodeToIds()` is safe to call concurrently.

**`Transform()`:** Our eager implementation reads all input rows sequentially, then processes batches sequentially. There's no concurrent access concern in the current design. If a future lazy cursor-based implementation is added, thread safety will need careful attention (see the custom-transformer-guide's note on `ThreadLocal<InferenceSession>`).

## The SchemaShape.Column Problem

ML.NET's `SchemaShape.Column` is a `readonly struct` with a *non-public constructor*. The `GetOutputSchema()` method on `IEstimator<T>` must return a `SchemaShape` containing these columns — but external code can't construct them through normal means.

Our workaround uses reflection:

```csharp
var colCtor = typeof(SchemaShape.Column).GetConstructors(
    BindingFlags.NonPublic | BindingFlags.Instance)[0];
var outputCol = (SchemaShape.Column)colCtor.Invoke([
    outputColumnName, VectorKind.Vector, NumberDataViewType.Single, false, null
]);
```

This is a known friction point for external ML.NET transform authors. It works reliably because the constructor signature has been stable across ML.NET versions, but it's another reason to eventually move to Approach D inside the ML.NET repo.

## The ICanSaveModel Requirement

`ITransformer` inherits from `ICanSaveModel`, which requires implementing `void Save(ModelSaveContext ctx)`. Since `ModelSaveContext` is part of ML.NET's internal serialization infrastructure and we can't meaningfully participate in it from external code, we throw `NotSupportedException`:

```csharp
void ICanSaveModel.Save(ModelSaveContext ctx)
{
    throw new NotSupportedException(
        "ML.NET native save is not supported. Use transformer.Save(path) instead.");
}
```

Users call `transformer.Save("path.mlnet")` instead of `mlContext.Model.Save(transformer, schema, "path")`. This is a minor API difference that would disappear if the transform moved into ML.NET.

## Why BertTokenizer (Not BPE)

The all-MiniLM-L6-v2 model (and most BERT-derived sentence-transformers) uses **WordPiece** tokenization, not BPE. The tokenizer vocabulary is distributed as `vocab.txt` — a simple newline-delimited file of tokens.

`Microsoft.ML.Tokenizers` v2.0.0 provides:
- `BertTokenizer.Create(Stream vocabStream)` — for WordPiece/BERT models
- `BpeTokenizer.Create(Stream vocab, Stream? merges)` — for GPT-2/BPE models

Our `LoadTokenizer()` currently supports `vocab.txt` files (BertTokenizer). Support for BPE tokenizers can be added by detecting the file format — see [extending.md](extending.md).
