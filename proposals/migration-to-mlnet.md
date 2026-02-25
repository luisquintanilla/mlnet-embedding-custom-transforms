# Migration to ML.NET (Approach C → Approach D)

## Overview

This document describes what changes when the modular transforms move from an external prototype (Approach C — direct `IEstimator`/`ITransformer` implementation) into the `dotnet/machinelearning` repository (Approach D — subclassing internal base classes).

**The core business logic carries over unchanged.** The migration is purely structural: replacing custom IDataView/cursor boilerplate with base class overrides, and replacing `ModelPackager` with native save/load.

## What Gets Deleted

| Approach C Boilerplate | Lines | Why It Exists |
|------------------------|-------|---------------|
| `TokenizerDataView : IDataView` | ~50 | Schema construction, cursor creation |
| `TokenizerCursor : DataViewRowCursor` | ~100 | Per-row tokenization, passthrough getters |
| `ScorerDataView : IDataView` | ~60 | Schema construction, cursor creation |
| `ScorerCursor : DataViewRowCursor` | ~200 | Lookahead batching, upstream caching, passthrough |
| `PoolerDataView : IDataView` | ~50 | Schema construction, cursor creation |
| `PoolerCursor : DataViewRowCursor` | ~100 | Per-row pooling, passthrough delegation |
| `ModelPackager.cs` | ~170 | Custom zip-based save/load |
| `SchemaShape.Column` reflection hack | ~10 | Non-public constructor workaround |
| **Total deleted** | **~740** | |

## What Gets Added

| Approach D Infrastructure | Lines | What It Does |
|---------------------------|-------|-------------|
| `[LoadableClass]` attributes (per transform) | ~4 each | Register transforms for save/load discovery |
| `SaveModel()` overrides (per transform) | ~20 each | Serialize state to `ModelSaveContext` |
| `static Create()` factories (per transform) | ~20 each | Deserialize from `ModelLoadContext` |
| `GetVersionInfo()` (per transform) | ~5 each | Version compatibility tracking |
| **Total added** | **~150** | |

**Net reduction: ~590 lines of boilerplate eliminated.**

## Per-Transform Migration

### TextTokenizerTransformer

```
BEFORE (Approach C):                        AFTER (Approach D):
─────────────────────                       ─────────────────────

TextTokenizerTransformer : ITransformer     TextTokenizerTransformer : OneToOneTransformerBase
├─ Transform() → returns TokenizerDataView  ├─ MakeRowMapper() → returns Mapper
├─ GetOutputSchema() manual                 │   (cursor creation is automatic)
├─ ICanSaveModel.Save() → throws           ├─ SaveModel(ModelSaveContext ctx)
│                                           │   ctx.SaveBinaryStream("Tokenizer", vocabBytes)
TokenizerDataView : IDataView              │   ctx.Writer.Write(options.MaxTokenLength)
├─ Schema (manual builder)                  │   ctx.Writer.Write(options.OutputTokenTypeIds)
├─ GetRowCursor() → TokenizerCursor        │
├─ GetRowCursorSet()                        ├─ static Create(IHostEnvironment, ModelLoadContext)
│                                           │   var vocabBytes = ctx.LoadBinaryStream("Tokenizer")
TokenizerCursor : DataViewRowCursor        │   ... reconstruct transformer
├─ MoveNext() → tokenize one row           │
├─ GetGetter<T>() → passthrough + computed  └─ Mapper : OneToOneMapperBase
├─ GetIdGetter()                               ├─ MakeGetter(DataViewRow input, int iinfo, ...)
├─ IsColumnActive()                            │   → ValueGetter that tokenizes current row
├─ Dispose()                                   │   (same logic as TokenizerCursor.MoveNext)
                                               ├─ GetOutputColumnsCore()
                                               │   → column definitions (TokenIds, AttentionMask, etc.)
                                               └─ SaveModel() → delegates to parent
```

**What carries over:** `Tokenize()` (direct face), `LoadTokenizer()`, `TokenizedBatch`, all tokenization math.

### OnnxTextModelScorerTransformer

```
BEFORE (Approach C):                        AFTER (Approach D):
─────────────────────                       ─────────────────────

OnnxTextModelScorerTransformer : ITransformer  OnnxTextModelScorerTransformer : RowToRowTransformerBase
├─ Transform() → returns ScorerDataView        ├─ MakeRowMapper() → returns Mapper
├─ RunOnnxBatch() (shared inference logic)     ├─ RunOnnxBatch() (unchanged — shared logic)
├─ Score() (direct face)                       ├─ Score() (direct face, unchanged)
├─ Dispose() → disposes InferenceSession       ├─ Dispose()
│                                              ├─ SaveModel(ModelSaveContext ctx)
ScorerDataView : IDataView                    │   ctx.SaveBinaryStream("Model", onnxModelBytes)
├─ Schema (manual builder)                     │   ctx.Writer.Write(options...)
├─ GetRowCursor() → ScorerCursor              │
├─ GetRowCursorSet()                           ├─ static Create(IHostEnvironment, ModelLoadContext)
│                                              │   var onnxBytes = ctx.LoadBinaryStream("Model")
ScorerCursor : DataViewRowCursor              │   ... reconstruct with InferenceSession
├─ MoveNext() → lookahead batch fill           │
├─ FillNextBatch() → read N rows + ONNX       └─ Mapper : MapperBase
├─ CacheUpstreamValues()                          ├─ MakeGetter(DataViewRow input, int iinfo, ...)
├─ GetGetter<T>() → cached results               │   → ValueGetter with lookahead batch cache
├─ GetCachedUpstreamGetter<T>()                   │   (same FillNextBatch logic as ScorerCursor)
├─ GetIdGetter()                                  ├─ GetOutputColumnsCore()
├─ Dispose()                                      ├─ GetDependenciesCore()
                                                  │   → token column dependencies
                                                  └─ SaveModel() → delegates to parent
```

**Key insight:** The lookahead batching logic in `MakeGetter()` is architecturally identical to the `ScorerCursor.FillNextBatch()` pattern. The getter maintains a batch cache, checks if it's exhausted, and refills by reading ahead from the input `DataViewRow`. The difference is that `MapperBase` provides the cursor and row lifecycle automatically — you just implement the getter delegate.

**What carries over:** `RunOnnxBatch()`, `Score()` (direct face), `DiscoverModelMetadata()`, `FindTensorName()` / `TryFindTensorName()`, `OnnxModelMetadata` record, all ONNX inference logic.

### EmbeddingPoolingTransformer

```
BEFORE (Approach C):                        AFTER (Approach D):
─────────────────────                       ─────────────────────

EmbeddingPoolingTransformer : ITransformer  EmbeddingPoolingTransformer : OneToOneTransformerBase
├─ Transform() → returns PoolerDataView     ├─ MakeRowMapper() → returns Mapper
├─ Pool() (direct face)                     ├─ Pool() (direct face, unchanged)
├─ PoolSingleRow() (cursor helper)          │
│                                           ├─ SaveModel(ModelSaveContext ctx)
PoolerDataView : IDataView                 │   ctx.Writer.Write((int)options.Pooling)
├─ Schema (manual builder)                  │   ctx.Writer.Write(options.HiddenDim)
├─ GetRowCursor() → PoolerCursor           │   ctx.Writer.Write(options.Normalize)
├─ GetRowCursorSet()                        │
│                                           ├─ static Create(IHostEnvironment, ModelLoadContext)
PoolerCursor : DataViewRowCursor           │
├─ MoveNext() → pool one row               └─ Mapper : OneToOneMapperBase
├─ GetGetter<T>() → passthrough + computed     ├─ MakeGetter(DataViewRow input, int iinfo, ...)
├─ GetIdGetter()                               │   → ValueGetter that pools current row
├─ Dispose()                                   │   (same logic as PoolerCursor.MoveNext)
                                               ├─ GetOutputColumnsCore()
                                               └─ SaveModel() → delegates to parent
```

**What carries over:** `Pool()` (direct face), `PoolSingleRow()`, all pooling math, `EmbeddingPooling.cs` (static math class, unchanged).

### OnnxTextEmbeddingEstimator (Facade)

```
BEFORE (Approach C):                        AFTER (Approach D):
─────────────────────                       ─────────────────────

OnnxTextEmbeddingEstimator                  OnnxTextEmbeddingEstimator
  : IEstimator<T>                             : IEstimator<T> (or custom base)
├─ Fit() → creates 3 sub-transforms        ├─ Fit() → creates 3 sub-transforms (unchanged)
├─ GetOutputSchema() → delegates            ├─ GetOutputSchema() → delegates (unchanged)
│                                           │
OnnxTextEmbeddingTransformer                OnnxTextEmbeddingTransformer
├─ Transform() → chains wrapping DataViews  ├─ Transform() → chains via TransformerChain
├─ GenerateEmbeddings() → direct faces      ├─ GenerateEmbeddings() → direct faces (unchanged)
├─ Save() → ModelPackager.Save()            ├─ Save() → mlContext.Model.Save() (native!)
├─ Load() → ModelPackager.Load()            ├─ Load() → mlContext.Model.Load() (native!)
                                            │
ModelPackager.cs (custom zip)               (DELETED — native save/load handles everything)
├─ Save() → zip with model.onnx, vocab.txt
├─ Load() → extract zip, reconstruct
```

**What carries over:** `Fit()` logic, `GenerateEmbeddings()` (direct face composition), `OnnxTextEmbeddingOptions`, all composition logic.

### OnnxEmbeddingGenerator

```
BEFORE (Approach C):                        AFTER (Approach D):
─────────────────────                       ─────────────────────

OnnxEmbeddingGenerator                      OnnxEmbeddingGenerator
  : IEmbeddingGenerator<string, Embedding<float>>    (UNCHANGED)
├─ GenerateAsync() → _transformer.GenerateEmbeddings()
├─ Metadata
├─ GetService<T>()
├─ Dispose()
```

**Completely unchanged.** The MEAI bridge doesn't depend on any internal plumbing.

## Save/Load: Custom Zip → Native ML.NET

### Current (Approach C)
```csharp
// Save
transformer.Save("my-model.mlnet");
// → ModelPackager creates zip: model.onnx + vocab.txt + config.json + manifest.json

// Load
var transformer = OnnxTextEmbeddingTransformer.Load(mlContext, "my-model.mlnet");
// → ModelPackager extracts zip, reconstructs via estimator.Fit()
```

### After Migration (Approach D)
```csharp
// Save — standard ML.NET API!
mlContext.Model.Save(transformer, dataView.Schema, "my-model.zip");
// → Each sub-transform's SaveModel() serializes to ModelSaveContext
//   Tokenizer: binary vocab stream + options
//   Scorer: binary ONNX model stream + options + tensor metadata
//   Pooler: options (pooling strategy, normalize, dims)

// Load — standard ML.NET API!
var transformer = mlContext.Model.Load("my-model.zip", out var schema);
// → [LoadableClass] + Create() factory reconstructs each sub-transform
```

This means:
- ✅ `mlContext.Model.Save()` / `mlContext.Model.Load()` works natively
- ✅ Interoperable with other ML.NET transforms in the same pipeline
- ✅ No custom `ModelPackager` needed
- ✅ `TransformerChain` serializes all transforms in sequence automatically

## Assembly-Level Attributes Required

Each transform needs `[LoadableClass]` registrations for the ML.NET loader discovery system:

```csharp
// One block per transform, at assembly level:

[assembly: LoadableClass(
    TextTokenizerTransformer.Summary,
    typeof(IDataTransform),
    typeof(TextTokenizerTransformer),
    null,
    typeof(SignatureLoadDataTransform),
    TextTokenizerTransformer.UserName,
    TextTokenizerTransformer.LoaderSignature)]

[assembly: LoadableClass(
    typeof(TextTokenizerTransformer),
    null,
    typeof(SignatureLoadModel),
    TextTokenizerTransformer.UserName,
    TextTokenizerTransformer.LoaderSignature)]

// ... similar for OnnxTextModelScorerTransformer, EmbeddingPoolingTransformer
```

## What the Prototype Validates

The Approach C prototype validates:

1. **API design** — Options classes, extension methods, column naming conventions
2. **Composition pattern** — Three-transform chain with facade
3. **Auto-discovery** — ONNX tensor metadata probing
4. **Lookahead batching** — Correct results with batch throughput
5. **Direct face pattern** — Efficient non-IDataView path for MEAI
6. **Task extensibility** — Shared tokenizer + scorer across future tasks

All of these carry over directly. The migration is plumbing, not design.

## Migration Checklist

- [ ] Move source into `dotnet/machinelearning/src/Microsoft.ML.OnnxEmbeddingTransformer/` (or similar)
- [ ] Change inheritance: `ITransformer` → `RowToRowTransformerBase` / `OneToOneTransformerBase`
- [ ] Implement `MakeRowMapper()` returning nested `Mapper` class
- [ ] Implement `Mapper.MakeGetter()` with per-transform logic (tokenize / infer / pool)
- [ ] Implement `Mapper.GetOutputColumnsCore()` for schema definitions
- [ ] Implement `SaveModel()` / `Create()` on each transform
- [ ] Add `[LoadableClass]` assembly attributes
- [ ] Change estimators to subclass `TrivialEstimator<T>` where appropriate
- [ ] Delete `ModelPackager.cs`
- [ ] Delete all custom `IDataView` / `DataViewRowCursor` classes
- [ ] Delete `SchemaShape.Column` reflection workaround
- [ ] Update facade to use `TransformerChain<ITransformer>` (following `TextFeaturizingEstimator` pattern)
- [ ] Add NuGet package dependencies: `Microsoft.ML.OnnxRuntime`, `Microsoft.ML.Tokenizers`
- [ ] Add unit tests following ML.NET test conventions
- [ ] Update public API surface documentation
