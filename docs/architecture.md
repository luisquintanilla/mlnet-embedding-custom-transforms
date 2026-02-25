# Architecture

This document walks through every component in the solution and traces the data flow from raw text to final embedding vector. Code references point to the actual source files.

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Code                                       │
│                                                                         │
│   var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);   │
│   var transformer = estimator.Fit(data);                                │
│   var result = transformer.Transform(data);                             │
│                                                                         │
│   // OR via MEAI:                                                       │
│   IEmbeddingGenerator<string, Embedding<float>> gen =                  │
│       new OnnxEmbeddingGenerator(mlContext, transformer);              │
│   var embeddings = await gen.GenerateAsync(texts);                     │
└────────────────────┬───────────────────────┬────────────────────────────┘
                     │                       │
        ┌────────────▼──────────┐  ┌─────────▼──────────────┐
        │ OnnxTextEmbedding-    │  │ OnnxEmbeddingGenerator  │
        │ Estimator             │  │                         │
        │                       │  │ IEmbeddingGenerator     │
        │ IEstimator<T>         │  │ <string, Embedding<T>>  │
        │ • Fit() → Transformer │  │ • GenerateAsync()       │
        │ • GetOutputSchema()   │  │ • wraps Transformer     │
        │ • DiscoverMetadata()  │  └─────────┬───────────────┘
        │ • LoadTokenizer()     │            │
        └────────────┬─────────┘            │
                     │ creates              │ delegates to
                     ▼                      ▼
        ┌──────────────────────────────────────────────┐
        │ OnnxTextEmbeddingTransformer                 │
        │                                              │
        │ ITransformer, IDisposable                    │
        │ • Transform(IDataView) → IDataView           │
        │ • GenerateEmbeddings(texts) → float[][]      │
        │ • Save(path) / Load(ctx, path)               │
        │                                              │
        │ Holds:                                       │
        │  ├─ InferenceSession (ONNX model)            │
        │  ├─ Tokenizer (BertTokenizer)                │
        │  ├─ Discovered tensor names & dimensions     │
        │  └─ OnnxTextEmbeddingOptions                 │
        └──────────┬──────────────┬────────────────────┘
                   │              │
          uses     │              │ uses
                   ▼              ▼
        ┌────────────────┐  ┌──────────────┐
        │ EmbeddingPooling│  │ ModelPackager │
        │                │  │              │
        │ • Pool()       │  │ • Save()     │
        │ • ExtractPooled│  │ • Load()     │
        │ • MeanPool     │  │              │
        │ • ClsPool      │  │ ZIP:         │
        │ • MaxPool      │  │ model.onnx   │
        │ • L2Normalize  │  │ vocab.txt    │
        │                │  │ config.json  │
        │ TensorPrimitives│  │ manifest.json│
        └────────────────┘  └──────────────┘
```

## The 6-Stage Pipeline

When `transformer.Transform(dataView)` is called, data flows through 6 stages:

### Stage 1: Text Extraction from IDataView

The transformer reads the input text column from the `IDataView` using a cursor:

```csharp
// OnnxTextEmbeddingTransformer.cs — ReadTextColumn()
var col = dataView.Schema[_options.InputColumnName];
using var cursor = dataView.GetRowCursor(new[] { col });
var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);

while (cursor.MoveNext())
{
    getter(ref value);
    texts.Add(value.ToString());
}
```

This materializes all text values into a `List<string>`. This is the "eager" aspect of our Approach C implementation — all rows are read before any inference happens. The texts are then processed in batches of `BatchSize` (default: 32).

### Stage 2: Tokenization

For each batch, the `BertTokenizer` encodes each text into token IDs. We simultaneously build the `attention_mask` (1 for real tokens, 0 for padding):

```csharp
// OnnxTextEmbeddingTransformer.cs — ProcessBatch()
var idsTensor = Tensor.Create<long>(idsArray, [batchSize, seqLen]);
var maskTensor = Tensor.Create<long>(maskArray, [batchSize, seqLen]);

for (int b = 0; b < batchSize; b++)
{
    var tokens = _tokenizer.EncodeToIds(texts[b], seqLen, out _, out _);
    for (int s = 0; s < tokens.Count && s < seqLen; s++)
    {
        idsTensor[b, s] = tokens[s];   // Tensor<T> multi-dim indexing
        maskTensor[b, s] = 1;
    }
}
```

**Key detail:** `Tensor.Create<long>(idsArray, shape)` wraps the existing flat array without copying. The `[b, s]` indexing writes through to the backing array. This means `idsArray` and `maskArray` are ready for OrtValue creation without any data movement. See [tensor-deep-dive.md](tensor-deep-dive.md) for details.

### Stage 3: OrtValue Creation (Zero-Copy Bridge)

The flat backing arrays are passed directly to OnnxRuntime:

```csharp
var inputs = new Dictionary<string, OrtValue>
{
    [_inputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
    [_attentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
};
```

`OrtValue.CreateTensorValueFromMemory` pins the managed array and creates a native tensor that references it directly — no copy. The tensor names (`_inputIdsName`, `_attentionMaskName`) were auto-discovered from ONNX metadata during `Fit()`.

If the model has a `token_type_ids` input (detected during auto-discovery), a zero-filled array is also provided.

### Stage 4: ONNX Inference

```csharp
using var results = _session.Run(new RunOptions(), inputs, [_outputTensorName]);
var output = results[0];
var outputSpan = output.GetTensorDataAsSpan<float>();
```

The output is a flat `ReadOnlySpan<float>` with shape `[batchSize, seqLen, hiddenDim]` (for unpooled models) or `[batchSize, hiddenDim]` (for pre-pooled models). We read it directly from native memory — another zero-copy operation.

### Stage 5: Pooling and Normalization

This is where `TensorPrimitives` does the heavy lifting. The pooling strategy determines how per-token hidden states are reduced to a single vector:

```csharp
// EmbeddingPooling.cs — dispatches by strategy
if (_modelHasPooledOutput)
    return EmbeddingPooling.ExtractPooled(outputSpan, batchSize, _hiddenDim, _options.Normalize);
else
    return EmbeddingPooling.Pool(outputSpan, maskArray, batchSize, seqLen, _hiddenDim,
        _options.Pooling, _options.Normalize);
```

For mean pooling specifically, the math is:

```
embedding[d] = Σ (hidden_state[s, d] × attention_mask[s]) / Σ attention_mask[s]
               s                                            s
```

Implemented with SIMD-accelerated `TensorPrimitives.Add` and `TensorPrimitives.Divide`. See [tensor-deep-dive.md](tensor-deep-dive.md) for the full walkthrough.

### Stage 6: Output Assembly

The final embeddings are assembled into an output `IDataView`:

**For ML.NET pipeline usage:**
```csharp
// BuildOutputDataView creates rows with Text + Embedding columns
_mlContext.Data.LoadFromEnumerable(rows);
```

**For MEAI usage:**
```csharp
// OnnxEmbeddingGenerator.cs — wraps float[] into Embedding<float>
var result = new GeneratedEmbeddings<Embedding<float>>(
    embeddings.Select(e => new Embedding<float>(e)));
```

No data copy — `Embedding<float>` wraps the existing `float[]` array.

## Estimator Lifecycle: What Happens in `Fit()`

The estimator is *trivial* — there's nothing to learn from training data. `Fit()` performs validation and initialization:

```
Fit(IDataView input)
  │
  ├─ 1. Validate input schema has the text column
  │     schema.GetColumnOrNull(inputColumnName) != null
  │
  ├─ 2. Create InferenceSession from ONNX model file
  │     new InferenceSession(options.ModelPath)
  │
  ├─ 3. Auto-discover tensor metadata
  │     DiscoverModelMetadata(session)
  │     ├─ Probe InputMetadata for input_ids, attention_mask, token_type_ids
  │     ├─ Probe OutputMetadata for sentence_embedding or last_hidden_state
  │     ├─ Determine if model has pre-pooled output
  │     └─ Extract embedding dimension from output tensor shape
  │
  ├─ 4. Load tokenizer
  │     BertTokenizer.Create(stream) from vocab.txt
  │
  └─ 5. Return OnnxTextEmbeddingTransformer with all discovered state
```

The `GetOutputSchema()` method also probes the ONNX model (it creates a temporary `InferenceSession` to read the embedding dimension), so the schema is accurate even before `Fit()` is called. This allows ML.NET pipeline validation to work correctly.

## MEAI Bridge: OnnxEmbeddingGenerator

The MEAI wrapper provides a clean `IEmbeddingGenerator<string, Embedding<float>>` interface. Its implementation is thin — it delegates to the transformer's internal `GenerateEmbeddings()` method which bypasses `IDataView` entirely:

```
GenerateAsync(IEnumerable<string> values)
  │
  ├─ Convert to IReadOnlyList<string>
  │
  ├─ Call transformer.GenerateEmbeddings(textList)
  │   └─ Same ProcessBatch() pipeline as Transform()
  │      but skips IDataView input/output overhead
  │
  └─ Wrap float[][] into GeneratedEmbeddings<Embedding<float>>
```

This allows:
- Using the same model via ML.NET pipelines (IDataView world)
- Using it via MEAI for RAG, search, or any `IEmbeddingGenerator` consumer
- Swapping between providers (OpenAI ↔ local ONNX) via the MEAI interface

The generator also supports ownership semantics: if constructed with `ownsTransformer: true` (or from a model path), it disposes the transformer when the generator is disposed.

## Save/Load Mechanics

### Saving

`ModelPackager.Save()` creates a zip archive:

```
transformer.Save("my-model.mlnet")
  │
  ├─ Create ZipArchive
  ├─ Copy model.onnx from ModelPath → archive
  ├─ Copy vocab.txt from TokenizerPath → archive
  ├─ Serialize OnnxTextEmbeddingOptions → config.json → archive
  └─ Write manifest.json (version, embedding dim, timestamp)
```

The tokenizer file is stored with its original filename (e.g., `vocab.txt`), and that filename is recorded in `config.json` so the loader knows what to look for.

### Loading

`ModelPackager.Load()` reconstructs the transformer:

```
OnnxTextEmbeddingTransformer.Load(mlContext, "my-model.mlnet")
  │
  ├─ Extract zip to temp directory
  ├─ Read config.json → OnnxTextEmbeddingOptions
  ├─ Set ModelPath = extracted model.onnx
  ├─ Set TokenizerPath = extracted vocab.txt
  ├─ Create OnnxTextEmbeddingEstimator with reconstructed options
  ├─ Create dummy IDataView for schema validation
  └─ Call estimator.Fit(dummyData) → new transformer
         └─ This runs full auto-discovery again on the extracted model
```

The round-trip is bit-perfect — the same ONNX bytes and vocab file produce identical embeddings.
