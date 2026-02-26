# Architecture

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                        │
│                                                                              │
│  // Composable pipeline (new):                                               │
│  var pipeline = mlContext.Transforms.TokenizeText(tokenizerOpts)             │
│      .Append(mlContext.Transforms.ScoreOnnxTextEmbedding(scorerOpts))            │
│      .Append(mlContext.Transforms.PoolEmbedding(poolingOpts));               │
│                                                                              │
│  // Convenience API (unchanged):                                             │
│  var estimator = mlContext.Transforms.OnnxTextEmbedding(options);            │
│                                                                              │
│  // MEAI usage (unchanged):                                                  │
│  IEmbeddingGenerator<string, Embedding<float>> gen = ...;                   │
│  var embeddings = await gen.GenerateAsync(texts);                            │
│                                                                              │
│  // Provider-agnostic ML.NET transform (new):                                │
│  var estimator = mlContext.Transforms.TextEmbedding(generator);             │
└──────────────┬────────────────────────────────┬──────────────────────────────┘
               │                                │
   ┌───────────▼──────────────┐     ┌───────────▼─────────────────────┐
   │ OnnxTextEmbedding-       │     │ EmbeddingGenerator-             │
   │ Estimator (facade)       │     │ Estimator (new)                 │
   │                          │     │                                 │
   │ Chains 3 transforms      │     │ Wraps IEmbeddingGenerator       │
   │ internally               │     │ Provider-agnostic               │
   │                          │     │ Text col → Embedding col        │
   │ Returns composite        │     │                                 │
   │ OnnxTextEmbedding-       │     │ Works with:                     │
   │ Transformer              │     │ • OnnxEmbeddingGenerator        │
   └───────────┬──────────────┘     │ • OpenAI / Azure / any MEAI     │
               │                    └─────────────────────────────────┘
               │ chains
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│             Reusable Foundation (any transformer ONNX model)                  │
│                                                                              │
│  ┌────────────────────┐     ┌──────────────────────────────┐                 │
│  │ TextTokenizer-     │     │ OnnxTextEmbeddingScorer-         │                 │
│  │ Transformer        │     │ Transformer                  │                 │
│  │                    │     │                              │                 │
│  │ Text →             │     │ TokenIds + AttentionMask +   │                 │
│  │   TokenIds         │────▶│ TokenTypeIds →               │                 │
│  │   AttentionMask    │     │   RawOutput                  │                 │
│  │   TokenTypeIds     │     │                              │                 │
│  │                    │     │ Wraps InferenceSession        │                 │
│  │ Wraps              │     │ Auto-discovers tensor names   │                 │
│  │ BertTokenizer      │     │ Handles batching              │                 │
│  │ (extensible)       │     │ Task-agnostic                 │                 │
│  └────────────────────┘     └──────────────┬───────────────┘                 │
│                                            │                                 │
└────────────────────────────────────────────┼─────────────────────────────────┘
                                             │
                  ┌──────────────────────────┬┼──────────────────────┐
                  │                          ││                      │
                  ▼                          ▼│                      ▼
  ┌───────────────────────┐  ┌───────────────▼──────┐  ┌────────────────────┐
  │ EmbeddingPooling-     │  │ Softmax-             │  │ NerDecoding-       │
  │ Transformer           │  │ Transformer          │  │ Transformer        │
  │                       │  │ (future)             │  │ (future)           │
  │ RawOutput +           │  │                      │  │                    │
  │ AttentionMask →       │  │ logits →             │  │ per-token logits → │
  │   Embedding           │  │ class probabilities  │  │ entity spans       │
  │                       │  │                      │  │                    │
  │ • Mean/CLS/Max pool   │  │                      │  │                    │
  │ • L2 normalize        │  │                      │  │                    │
  └───────────────────────┘  └──────────────────────┘  └────────────────────┘
```

## IDataView Column Flow

```
Input IDataView:
  │ Text (string, TextDataViewType)
  ▼
TextTokenizerTransformer:
  │ Text (string)                       ← passed through
  │ TokenIds (VBuffer<long>)            ← NEW: padded to MaxTokenLength
  │ AttentionMask (VBuffer<long>)       ← NEW: 1=real token, 0=padding
  │ TokenTypeIds (VBuffer<long>)        ← NEW: zeros (or segment IDs for text pairs)
  ▼
OnnxTextEmbeddingScorerTransformer:
  │ Text (string)                       ← passed through
  │ TokenIds (VBuffer<long>)            ← passed through
  │ AttentionMask (VBuffer<long>)       ← passed through
  │ TokenTypeIds (VBuffer<long>)        ← passed through
  │ RawOutput (VBuffer<float>)          ← NEW: shape depends on model:
  │                                         [hiddenDim] if pre-pooled
  │                                         [seqLen × hiddenDim] if unpooled
  ▼
EmbeddingPoolingTransformer:
  │ Text (string)                       ← passed through
  │ Embedding (VBuffer<float>)          ← NEW: [hiddenDim], pooled + normalized
  ▼
Output IDataView
```

## Lazy Evaluation via Custom IDataView / Cursor

Each transform returns a **wrapping IDataView** from `Transform()` — no data is materialized. Computation happens lazily when a downstream consumer iterates via a cursor.

```csharp
// Transform() does NO work — just wraps
public IDataView Transform(IDataView input)
{
    return new TokenizerDataView(input, _tokenizer, _options);
}
```

When the final consumer iterates, cursors chain upstream:

```
PoolerCursor.MoveNext()
  → ScorerCursor.MoveNext()
      → TokenizerCursor.MoveNext()
          → InputCursor.MoveNext()
```

At any given moment, only **one batch** of intermediate data exists in memory (~6 MB for a batch of 32 with a 384-dim model). The 1.9 GB intermediate materialization problem is eliminated.

### Lookahead Batching (Scorer Only)

The tokenizer and pooler are cheap (microseconds per row) — they process row-by-row. The ONNX scorer is expensive and batch-sensitive (15x faster at batch=32 vs. batch=1). The scorer cursor implements **lookahead batching**:

```
ScorerCursor.MoveNext():
  if cached batch exhausted:
    read next 32 rows from upstream tokenizer cursor
    pack token arrays into flat batch tensors
    single session.Run() call
    cache 32 output arrays
  return cached result[batchIndex++]
```

This gives us batch throughput with lazy memory semantics.

### Per-Transform Boilerplate

Each lazy transform requires three types:

| Type | Purpose | Lines (est.) |
|------|---------|-------------|
| `XxxDataView : IDataView` | Wraps upstream IDataView, adds output column(s) to schema | ~50 |
| `XxxCursor : DataViewRowCursor` | Chains to upstream cursor, computes values on MoveNext | ~80-150 |
| Column getter delegates | `ValueGetter<VBuffer<T>>` that return computed values | ~20 |

This boilerplate is **temporary scaffolding** — when migrating to ML.NET (Approach D), it's replaced by `RowToRowTransformerBase` / `MapperBase` which provide cursor and schema infrastructure automatically. See [migration-to-mlnet.md](migration-to-mlnet.md).

## Internal vs. External Composition

Each transform exposes TWO faces:

### ML.NET Face (IDataView-based, Lazy)
Used by ML.NET pipelines. `Transform()` returns a wrapping IDataView. Computation happens lazily when a cursor iterates. No materialization.

### Direct Face (List/Array-based, Eager)
Used internally by the facade's `GenerateEmbeddings()` and the MEAI generator. Bypasses IDataView entirely for zero-overhead batch processing.

```csharp
// ML.NET face (public) — lazy, wraps input
public IDataView Transform(IDataView input) { ... }

// Direct face (internal) — eager, processes batch directly
internal TokenizedBatch Tokenize(IReadOnlyList<string> texts) { ... }
internal float[][] Score(TokenizedBatch batch) { ... }
internal float[][] Pool(float[][] rawOutput, long[][] attentionMasks) { ... }
```

The direct faces are used by `GenerateEmbeddings()` and `OnnxEmbeddingGenerator` for maximum throughput without IDataView overhead.

## Save/Load Strategy

The composite `OnnxTextEmbeddingTransformer` (facade) saves/loads as a single zip (same as today):

```
embedding-model.mlnet (zip)
├── model.onnx
├── vocab.txt
├── config.json        ← includes all three transforms' options
└── manifest.json
```

Individual transforms don't need standalone save/load — they're reconstructed from the facade's saved state.

The provider-agnostic `EmbeddingGeneratorTransformer` does NOT support save/load natively (since `IEmbeddingGenerator` has no save contract). ONNX-backed generators can be saved/loaded via `OnnxEmbeddingGenerator.Save()/Load()`.
