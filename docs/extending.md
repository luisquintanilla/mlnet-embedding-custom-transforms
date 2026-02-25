# Extending

This document covers how to modify, extend, and harden the solution. Each section includes the specific files to change and the patterns to follow.

## Adding New Pooling Strategies

**File to modify:** `EmbeddingPooling.cs`

1. Add a value to the `PoolingStrategy` enum in `PoolingStrategy.cs`:

```csharp
public enum PoolingStrategy
{
    MeanPooling,
    ClsToken,
    MaxPooling,
    WeightedMeanPooling  // ← new
}
```

2. Add a private method and a case to the `Pool()` switch in `EmbeddingPooling.cs`:

```csharp
PoolingStrategy.WeightedMeanPooling => WeightedMeanPool(
    hiddenStates, attentionMask, b, seqLen, hiddenDim),
```

3. Implement the method following the existing pattern — use `TensorPrimitives` for SIMD math:

```csharp
private static float[] WeightedMeanPool(
    ReadOnlySpan<float> hiddenStates,
    ReadOnlySpan<long> attentionMask,
    int batchIdx, int seqLen, int hiddenDim)
{
    var embedding = new float[hiddenDim];
    float weightSum = 0;

    for (int s = 0; s < seqLen; s++)
    {
        if (attentionMask[batchIdx * seqLen + s] > 0)
        {
            float weight = (float)(s + 1) / seqLen;  // linear position weight
            int offset = (batchIdx * seqLen + s) * hiddenDim;
            ReadOnlySpan<float> tokenEmbed = hiddenStates.Slice(offset, hiddenDim);

            // Scale and accumulate
            var scaled = new float[hiddenDim];
            TensorPrimitives.Multiply(tokenEmbed, weight, scaled);
            TensorPrimitives.Add(embedding, scaled, embedding);
            weightSum += weight;
        }
    }

    if (weightSum > 0)
        TensorPrimitives.Divide(embedding, weightSum, embedding);

    return embedding;
}
```

## Supporting New Tokenizer Formats

**File to modify:** `OnnxTextEmbeddingEstimator.cs` — `LoadTokenizer()` method

Currently only `vocab.txt` (BertTokenizer/WordPiece) is supported. To add BPE support:

```csharp
internal static Tokenizer LoadTokenizer(string path)
{
    var ext = Path.GetExtension(path).ToLowerInvariant();
    var fileName = Path.GetFileName(path).ToLowerInvariant();

    return (ext, fileName) switch
    {
        (".txt", _) => BertTokenizer.Create(File.OpenRead(path)),

        // BPE tokenizer (GPT-2 style) — requires vocab.json + merges.txt
        (".json", "vocab.json") => LoadBpeTokenizer(path),

        _ => throw new NotSupportedException(
            $"Unsupported tokenizer file: '{fileName}'. " +
            $"Use vocab.txt for BERT/WordPiece or vocab.json for BPE.")
    };
}

private static Tokenizer LoadBpeTokenizer(string vocabPath)
{
    var dir = Path.GetDirectoryName(vocabPath)!;
    var mergesPath = Path.Combine(dir, "merges.txt");

    using var vocabStream = File.OpenRead(vocabPath);
    using var mergesStream = File.Exists(mergesPath)
        ? File.OpenRead(mergesPath) : null;

    return BpeTokenizer.Create(vocabStream, mergesStream);
}
```

**Also update `ModelPackager`:** When saving, you'd need to bundle both `vocab.json` and `merges.txt` for BPE models.

## Using Different ONNX Models

The transform is model-agnostic. Any ONNX model that follows the sentence-transformer convention works:

### Model Requirements

| Input | Type | Shape | Required? |
|-------|------|-------|:---------:|
| `input_ids` | int64 | `[batch, seq_len]` | ✅ |
| `attention_mask` | int64 | `[batch, seq_len]` | ✅ |
| `token_type_ids` | int64 | `[batch, seq_len]` | ❌ auto-detected |

| Output | Type | Shape | Behavior |
|--------|------|-------|----------|
| `last_hidden_state` | float32 | `[batch, seq_len, hidden_dim]` | Pooling applied |
| `sentence_embedding` | float32 | `[batch, hidden_dim]` | Used directly |

### Non-Standard Models

If tensor names differ from the convention, use the override options:

```csharp
var options = new OnnxTextEmbeddingOptions
{
    ModelPath = "custom-model.onnx",
    TokenizerPath = "vocab.txt",
    InputIdsName = "tokens",              // override
    AttentionMaskName = "mask",           // override
    OutputTensorName = "embeddings",      // override
};
```

### Exporting Models to ONNX

If a model isn't available in ONNX format, export it with `optimum-cli`:

```bash
pip install optimum[exporters]
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 ./onnx-model/
```

This creates `model.onnx` along with `vocab.txt` (or `tokenizer.json` depending on the model).

## Path to Lazy Cursor-Based Evaluation

The current implementation is eager — `Transform()` materializes all rows before returning. A lazy implementation would compute embeddings on-demand as a cursor advances:

### What It Would Look Like

```csharp
public IDataView Transform(IDataView input)
{
    // Return a wrapping IDataView, don't materialize anything
    return new EmbeddingDataView(input, this);
}
```

Where `EmbeddingDataView` implements:

```csharp
class EmbeddingDataView : IDataView
{
    public DataViewSchema Schema { get; }  // input schema + embedding column
    public bool CanShuffle => false;
    public long? GetRowCount() => _input.GetRowCount();

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columns, Random? rand = null)
    {
        return new EmbeddingCursor(_input, _transformer, columns);
    }
}
```

And `EmbeddingCursor` computes embeddings when the getter is invoked:

```csharp
class EmbeddingCursor : DataViewRowCursor
{
    private readonly DataViewRowCursor _inputCursor;
    private float[]? _currentEmbedding;

    // Getter registered for the embedding column
    private void GetEmbedding(ref VBuffer<float> value)
    {
        if (_currentEmbedding == null)
        {
            // Compute embedding for current row
            var text = GetCurrentText();
            _currentEmbedding = _transformer.GenerateEmbeddings([text])[0];
        }
        value = new VBuffer<float>(_currentEmbedding.Length, _currentEmbedding);
    }
}
```

### Challenges

1. **Per-row inference is slow.** ONNX models are optimized for batches. A single-row batch is 5-10x slower per item than a batch of 32.
2. **Lookahead batching** is complex — the cursor would need to read ahead N rows, batch-infer them, then serve results as the cursor advances.
3. **Thread safety** — ML.NET cursors can be created concurrently. Each would need its own `InferenceSession` or a session pool.
4. **Schema construction** — `DataViewSchema.Builder` and cursor registration require careful implementation to satisfy ML.NET's contracts.

**Recommendation:** The eager batched approach is sufficient for most embedding workloads. Lazy evaluation is worthwhile only for very large datasets that don't fit in memory.

## Path to Approach D: Inside ML.NET

If this transform is adopted into `dotnet/machinelearning`, it would use the internal base classes for full ML.NET integration:

### Changes Required

1. **Subclass `RowToRowTransformerBase`** instead of implementing `ITransformer` directly:

```csharp
public sealed class OnnxTextEmbeddingTransformer
    : RowToRowTransformerBase  // ← private protected constructor accessible from within ML.NET
{
    private protected override IRowMapper MakeRowMapper(DataViewSchema inputSchema)
    {
        return new OnnxEmbeddingMapper(this, inputSchema);
    }
}
```

2. **Implement `MapperBase`** for lazy cursor-based evaluation:

```csharp
private sealed class OnnxEmbeddingMapper : MapperBase
{
    protected override Delegate MakeGetter(DataViewRow input, int iinfo, ...)
    {
        // Return a delegate that computes the embedding for the current row
        ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) => { ... };
        return getter;
    }
}
```

3. **Add `[LoadableClass]` attributes** for native save/load:

```csharp
[assembly: LoadableClass(typeof(OnnxTextEmbeddingTransformer), null,
    typeof(SignatureLoadModel), "ONNX Text Embedding", LoaderSignature)]
```

4. **Implement `SaveModel()`** with `ModelSaveContext`:

```csharp
private protected override void SaveModel(ModelSaveContext ctx)
{
    ctx.SetVersionInfo(GetVersionInfo());
    ctx.SaveBinaryStream("Model", w => { /* write ONNX bytes */ });
    ctx.SaveBinaryStream("Tokenizer", w => { /* write vocab bytes */ });
    ctx.Writer.Write(_options.MaxTokenLength);
    ctx.Writer.Write((int)_options.Pooling);
    // ... write all config
}
```

5. **Add static `Create()` factory** for loading:

```csharp
private static OnnxTextEmbeddingTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
{
    ctx.CheckVersionInfo(GetVersionInfo());
    var modelBytes = ctx.LoadBinaryStream("Model");
    var vocabBytes = ctx.LoadBinaryStream("Tokenizer");
    // ... read config, construct transformer
}
```

This would give full `mlContext.Model.Save()` / `mlContext.Model.Load()` support and lazy cursor-based evaluation through `MapperBase`.

## Production Hardening

### GPU Support

OnnxRuntime supports GPU execution via CUDA or DirectML. Add GPU options:

```csharp
var sessionOptions = new SessionOptions();
if (options.GpuDeviceId.HasValue)
{
    sessionOptions.AppendExecutionProvider_CUDA(options.GpuDeviceId.Value);
    // OR: sessionOptions.AppendExecutionProvider_DML(options.GpuDeviceId.Value);
}
var session = new InferenceSession(options.ModelPath, sessionOptions);
```

### Session Pooling

For high-throughput scenarios, a pool of `InferenceSession` instances can be used to parallelize inference across CPU threads:

```csharp
private readonly ObjectPool<InferenceSession> _sessionPool;
```

### Error Handling

The prototype has minimal error handling. Production code should:
- Validate tensor dimensions match expected shapes after ONNX run
- Handle OnnxRuntime exceptions (OOM, invalid model, etc.) gracefully
- Add telemetry / logging for inference timing
- Validate tokenizer output length doesn't exceed model's max sequence length

### Quantized Models

For deployment on edge devices, use quantized ONNX models (INT8 or FP16). The transform works without modification — OnnxRuntime handles quantized inference transparently. Just point `ModelPath` to the quantized `.onnx` file.
