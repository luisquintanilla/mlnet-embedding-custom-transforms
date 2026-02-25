# Architecture

This document walks through every component in the solution and traces the data flow from raw text to final embedding vector. Code references point to the actual source files.

## Component Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                        │
│                                                                              │
│  // Composable pipeline (new):                                               │
│  var pipeline = mlContext.Transforms.TokenizeText(tokenizerOpts)             │
│      .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))            │
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
│  │ TextTokenizer-     │     │ OnnxTextModelScorer-         │                 │
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
  │ TokenTypeIds (VBuffer<long>)        ← NEW: zeros (segment IDs)
  ▼
OnnxTextModelScorerTransformer:
  │ Text (string)                       ← passed through
  │ TokenIds (VBuffer<long>)            ← passed through
  │ AttentionMask (VBuffer<long>)       ← passed through
  │ TokenTypeIds (VBuffer<long>)        ← passed through
  │ RawOutput (VBuffer<float>)          ← NEW: shape depends on model
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

At any given moment, only **one batch** of intermediate data exists in memory (~6 MB for a batch of 32 with a 384-dim model).

### Lookahead Batching (Scorer Only)

The tokenizer and pooler are cheap (microseconds per row) — they process row-by-row. The ONNX scorer uses **lookahead batching**: it reads N rows from the upstream tokenizer cursor, packs them into a single ONNX batch, runs inference once, then serves cached results one at a time. This gives batch throughput with lazy memory semantics.

### Two Faces: ML.NET + Direct

Each transform exposes two faces:

- **ML.NET face** (`Transform(IDataView)`): Lazy, wraps input. Used by ML.NET pipelines.
- **Direct face** (`Tokenize()`, `Score()`, `Pool()`): Eager, processes batches directly. Used by `GenerateEmbeddings()` and `OnnxEmbeddingGenerator`.

## Estimator Lifecycle: What Happens in `Fit()`

The facade estimator (`OnnxTextEmbeddingEstimator`) chains three sub-estimators:

```
Fit(IDataView input)
  │
  ├─ 1. Create TextTokenizerEstimator → Fit → TextTokenizerTransformer
  │     Loads BertTokenizer from vocab.txt
  │
  ├─ 2. Create OnnxTextModelScorerEstimator → Fit → OnnxTextModelScorerTransformer
  │     Creates InferenceSession, auto-discovers tensor metadata
  │
  ├─ 3. Create EmbeddingPoolingEstimator → Fit → EmbeddingPoolingTransformer
  │     Auto-configured from scorer metadata (HiddenDim, IsPrePooled)
  │
  └─ 4. Return OnnxTextEmbeddingTransformer wrapping all three
```

## MEAI Bridge: OnnxEmbeddingGenerator

The MEAI wrapper delegates to `GenerateEmbeddings()`, which chains the three sub-transforms' **direct faces**:

```
GenerateEmbeddings(texts)
  │
  ├─ _tokenizer.Tokenize(batch) → TokenizedBatch
  ├─ _scorer.Score(batch) → float[][] (raw ONNX output)
  └─ _pooler.Pool(scored, attentionMasks) → float[][] (pooled embeddings)
```

## Save/Load Mechanics

The composite `OnnxTextEmbeddingTransformer` saves/loads as a single zip (same as before):

```
embedding-model.mlnet (zip)
├── model.onnx
├── vocab.txt
├── config.json        ← includes all options
└── manifest.json
```

Individual transforms don't need standalone save/load — they're reconstructed from the facade's saved state. The `EmbeddingGeneratorTransformer` does NOT support save/load (since `IEmbeddingGenerator` has no save contract).
