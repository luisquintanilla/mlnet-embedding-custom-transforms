# Architecture

## Component Diagram

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
  │ TokenTypeIds (VBuffer<long>)        ← NEW: zeros (or segment IDs for text pairs)
  ▼
OnnxTextModelScorerTransformer:
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

## Internal vs. External Composition

Each transform exposes TWO faces:

### ML.NET Face (IDataView-based)
Used by ML.NET pipelines. Reads input columns via cursor, writes output columns via `LoadFromEnumerable`.

### Direct Face (List/Array-based)
Used internally by the facade and MEAI generator. Bypasses IDataView for zero-overhead composition.

```csharp
// ML.NET face (public)
public IDataView Transform(IDataView input) { ... }

// Direct face (internal)
internal TokenizedBatch Tokenize(IReadOnlyList<string> texts) { ... }
internal float[][] Score(TokenizedBatch batch) { ... }
internal float[][] Pool(float[][] rawOutput, long[][] attentionMasks) { ... }
```

The facade and MEAI generator use the direct faces, avoiding IDataView materialization overhead for the common case.

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
