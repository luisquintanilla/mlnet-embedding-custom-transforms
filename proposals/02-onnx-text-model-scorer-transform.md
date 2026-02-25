# Transform 2: OnnxTextModelScorerEstimator / OnnxTextModelScorerTransformer

## Purpose

Runs ONNX inference on tokenized text inputs. This is the **universal second step** for any transformer-based ONNX model task. It takes token columns (produced by `TextTokenizerTransformer`) and outputs the raw model tensor. It is intentionally **task-agnostic** — it doesn't know whether the output will be pooled into embeddings, softmaxed into class probabilities, or decoded into entity spans.

## Why "OnnxTextModelScorer" and not "OnnxScorer"?

- ML.NET already has `OnnxScoringEstimator` / `OnnxTransformer` in `Microsoft.ML.OnnxTransformer` — a general-purpose ONNX inference transform that takes arbitrary named columns.
- Our transform is specialized for **transformer-architecture text models**: it expects tokenized input (`input_ids`, `attention_mask`, `token_type_ids`) and auto-discovers tensor names using transformer-model conventions.
- "TextModel" makes clear this is for BERT/GPT-style models, not arbitrary ONNX (e.g., image classifiers, tabular models).

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.Embeddings.Onnx/OnnxTextModelScorerEstimator.cs` | Estimator + options class |
| `src/MLNet.Embeddings.Onnx/OnnxTextModelScorerTransformer.cs` | Transformer |

## Options Class

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// Configuration for the ONNX text model scorer transform.
/// Runs inference on a transformer-architecture ONNX model (BERT, MiniLM, etc.).
/// </summary>
public class OnnxTextModelScorerOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; set; }

    // --- Input column names (must match tokenizer output) ---

    /// <summary>Name of the input token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the input attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>
    /// Name of the input token type IDs column. Default: "TokenTypeIds".
    /// Set to null if the model doesn't use token type IDs.
    /// </summary>
    public string? TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    // --- Output ---

    /// <summary>Name of the output column for raw model output. Default: "RawOutput".</summary>
    public string OutputColumnName { get; set; } = "RawOutput";

    // --- Inference configuration ---

    /// <summary>
    /// Maximum sequence length. Must match the tokenizer's MaxTokenLength.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>Batch size for ONNX inference. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;

    // --- ONNX tensor name overrides (null = auto-detect) ---

    /// <summary>ONNX input tensor name for token IDs. Null = auto-detect ("input_ids").</summary>
    public string? InputIdsTensorName { get; set; }

    /// <summary>ONNX input tensor name for attention mask. Null = auto-detect ("attention_mask").</summary>
    public string? AttentionMaskTensorName { get; set; }

    /// <summary>ONNX input tensor name for token type IDs. Null = auto-detect ("token_type_ids" if present).</summary>
    public string? TokenTypeIdsTensorName { get; set; }

    /// <summary>
    /// ONNX output tensor name. Null = auto-detect.
    /// Auto-detection prefers "sentence_embedding" / "pooler_output" (pre-pooled),
    /// falls back to "last_hidden_state" / "output" (unpooled).
    /// </summary>
    public string? OutputTensorName { get; set; }
}
```

### Design Note: Tensor Name Options vs. Column Name Options

The options class distinguishes two naming layers:
- **Column names** (`TokenIdsColumnName`, etc.): IDataView column names that coordinate with the tokenizer transform
- **Tensor names** (`InputIdsTensorName`, etc.): ONNX graph tensor names for the native inference session

This separation is important because column names are a pipeline concern (how transforms communicate) while tensor names are a model concern (what the ONNX graph expects). Users override tensor names only for non-standard models; column names only when composing with differently-named upstream transforms.

## Estimator

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextModelScorerTransformer.
/// Fit() validates the input schema, loads the ONNX model, and auto-discovers tensor metadata.
/// </summary>
public sealed class OnnxTextModelScorerEstimator : IEstimator<OnnxTextModelScorerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;

    public OnnxTextModelScorerEstimator(MLContext mlContext, OnnxTextModelScorerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
    }

    public OnnxTextModelScorerTransformer Fit(IDataView input)
    {
        // Validate input schema has token columns
        ValidateColumn(input.Schema, _options.TokenIdsColumnName);
        ValidateColumn(input.Schema, _options.AttentionMaskColumnName);
        if (_options.TokenTypeIdsColumnName != null)
            ValidateColumn(input.Schema, _options.TokenTypeIdsColumnName);

        // Load ONNX model and auto-discover tensor metadata
        var session = new InferenceSession(_options.ModelPath);
        var metadata = DiscoverModelMetadata(session);

        return new OnnxTextModelScorerTransformer(
            _mlContext, _options, session, metadata);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input columns exist
        // Probe model to get output dimensions
        // Add output column (VBuffer<float>)
    }

    /// <summary>
    /// Auto-discovers ONNX tensor names and output shape.
    /// Extracted from existing OnnxTextEmbeddingEstimator.DiscoverModelMetadata().
    /// </summary>
    internal OnnxModelMetadata DiscoverModelMetadata(InferenceSession session)
    {
        var inputMeta = session.InputMetadata;
        var outputMeta = session.OutputMetadata;

        // Discover input tensor names (same logic as current implementation)
        string inputIdsName = _options.InputIdsTensorName
            ?? FindTensorName(inputMeta, ["input_ids"], "input_ids");
        string attentionMaskName = _options.AttentionMaskTensorName
            ?? FindTensorName(inputMeta, ["attention_mask"], "attention_mask");
        string? tokenTypeIdsName = _options.TokenTypeIdsTensorName
            ?? TryFindTensorName(inputMeta, ["token_type_ids"]);

        // Discover output tensor name
        // (same priority logic as current: prefer sentence_embedding, fall back to last_hidden_state)
        string outputName;
        bool hasPooledOutput;
        int hiddenDim;
        int outputRank;

        if (_options.OutputTensorName != null)
        {
            outputName = _options.OutputTensorName;
            var dims = outputMeta[outputName].Dimensions;
            hasPooledOutput = !dims.Contains(-1) && dims.Length == 2;
            hiddenDim = (int)dims.Last();
            outputRank = dims.Length;
        }
        else
        {
            var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
            if (pooledName != null)
            {
                outputName = pooledName;
                hasPooledOutput = true;
                hiddenDim = (int)outputMeta[pooledName].Dimensions.Last();
                outputRank = 2;
            }
            else
            {
                outputName = FindTensorName(outputMeta,
                    ["last_hidden_state", "output", "hidden_states"],
                    outputMeta.Keys.First());
                hasPooledOutput = false;
                hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
                outputRank = 3;
            }
        }

        if (hiddenDim <= 0)
            throw new InvalidOperationException(
                $"Could not determine hidden dimension from ONNX output '{outputName}'.");

        return new OnnxModelMetadata(
            inputIdsName, attentionMaskName, tokenTypeIdsName,
            outputName, hiddenDim, hasPooledOutput, outputRank);
    }

    // FindTensorName / TryFindTensorName — same as current implementation
}

/// <summary>
/// Discovered ONNX model tensor metadata. Immutable record.
/// </summary>
internal sealed record OnnxModelMetadata(
    string InputIdsName,
    string AttentionMaskName,
    string? TokenTypeIdsName,
    string OutputTensorName,
    int HiddenDim,
    bool HasPooledOutput,
    int OutputRank);
```

## Transformer

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET ITransformer that runs ONNX inference on tokenized text inputs.
/// Task-agnostic — outputs the raw model tensor for downstream post-processing.
/// </summary>
public sealed class OnnxTextModelScorerTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextModelScorerOptions _options;
    private readonly InferenceSession _session;
    private readonly OnnxModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    internal OnnxTextModelScorerOptions Options => _options;

    /// <summary>Hidden dimension of the model output.</summary>
    public int HiddenDim => _metadata.HiddenDim;

    /// <summary>Whether the model outputs pre-pooled embeddings (e.g., sentence_embedding).</summary>
    public bool HasPooledOutput => _metadata.HasPooledOutput;

    /// <summary>Auto-discovered ONNX metadata.</summary>
    internal OnnxModelMetadata Metadata => _metadata;

    internal OnnxTextModelScorerTransformer(
        MLContext mlContext,
        OnnxTextModelScorerOptions options,
        InferenceSession session,
        OnnxModelMetadata metadata)
    {
        _mlContext = mlContext;
        _options = options;
        _session = session;
        _metadata = metadata;
    }

    public IDataView Transform(IDataView input)
    {
        // 1. Read token columns from input
        var (tokenIds, attentionMasks, tokenTypeIds) = ReadTokenColumns(input);

        // 2. Run ONNX inference in batches
        var rawOutputs = Score(tokenIds, attentionMasks, tokenTypeIds);

        // 3. Build output IDataView with original columns + raw output column
        return BuildOutputDataView(input, rawOutputs);
    }

    /// <summary>
    /// Direct face: run ONNX inference on pre-tokenized input without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal float[][] Score(TokenizedBatch batch)
    {
        return Score(batch.TokenIds, batch.AttentionMasks, batch.TokenTypeIds);
    }

    private float[][] Score(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds)
    {
        int totalRows = tokenIds.Length;
        int batchSize = _options.BatchSize;
        int seqLen = _options.MaxTokenLength;
        var allOutputs = new List<float[]>(totalRows);

        for (int start = 0; start < totalRows; start += batchSize)
        {
            int count = Math.Min(batchSize, totalRows - start);
            var batchOutputs = ProcessBatch(
                tokenIds, attentionMasks, tokenTypeIds,
                start, count, seqLen);
            allOutputs.AddRange(batchOutputs);
        }

        return [.. allOutputs];
    }

    private float[][] ProcessBatch(
        long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds,
        int startIdx, int batchSize, int seqLen)
    {
        // Build flat arrays from per-row arrays for this batch
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _metadata.TokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        for (int b = 0; b < batchSize; b++)
        {
            Array.Copy(tokenIds[startIdx + b], 0, idsArray, b * seqLen, seqLen);
            Array.Copy(attentionMasks[startIdx + b], 0, maskArray, b * seqLen, seqLen);
            if (typeIdsArray != null && tokenTypeIds != null)
                Array.Copy(tokenTypeIds[startIdx + b], 0, typeIdsArray, b * seqLen, seqLen);
        }

        // Create OrtValues from flat backing arrays (zero-copy)
        var inputs = new Dictionary<string, OrtValue>
        {
            [_metadata.InputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_metadata.AttentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_metadata.TokenTypeIdsName != null && typeIdsArray != null)
            inputs[_metadata.TokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        try
        {
            using var results = _session.Run(new RunOptions(), inputs, [_metadata.OutputTensorName]);
            var output = results[0];
            var outputSpan = output.GetTensorDataAsSpan<float>();

            // Extract per-row output tensors
            var batchOutputs = new float[batchSize][];

            if (_metadata.HasPooledOutput)
            {
                // Pre-pooled: shape [batchSize, hiddenDim] → one float[hiddenDim] per row
                for (int b = 0; b < batchSize; b++)
                {
                    batchOutputs[b] = outputSpan.Slice(b * _metadata.HiddenDim, _metadata.HiddenDim).ToArray();
                }
            }
            else
            {
                // Unpooled: shape [batchSize, seqLen, hiddenDim] → one float[seqLen * hiddenDim] per row
                int rowSize = seqLen * _metadata.HiddenDim;
                for (int b = 0; b < batchSize; b++)
                {
                    batchOutputs[b] = outputSpan.Slice(b * rowSize, rowSize).ToArray();
                }
            }

            return batchOutputs;
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    private (long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds) ReadTokenColumns(IDataView input)
    {
        // Read VBuffer<long> columns from IDataView via cursor
        var tokenIdsList = new List<long[]>();
        var attentionMaskList = new List<long[]>();
        var tokenTypeIdsList = _options.TokenTypeIdsColumnName != null ? new List<long[]>() : null;

        var tokenIdsCol = input.Schema[_options.TokenIdsColumnName];
        var attMaskCol = input.Schema[_options.AttentionMaskColumnName];
        var typeIdsCol = _options.TokenTypeIdsColumnName != null
            ? input.Schema[_options.TokenTypeIdsColumnName] : default;

        var activeColumns = new List<DataViewSchema.Column> { tokenIdsCol, attMaskCol };
        if (typeIdsCol.Name != null)
            activeColumns.Add(typeIdsCol);

        using var cursor = input.GetRowCursor(activeColumns);
        var tokenIdsGetter = cursor.GetGetter<VBuffer<long>>(tokenIdsCol);
        var attMaskGetter = cursor.GetGetter<VBuffer<long>>(attMaskCol);
        var typeIdsGetter = typeIdsCol.Name != null
            ? cursor.GetGetter<VBuffer<long>>(typeIdsCol) : null;

        VBuffer<long> tokenIdsBuffer = default;
        VBuffer<long> attMaskBuffer = default;
        VBuffer<long> typeIdsBuffer = default;

        while (cursor.MoveNext())
        {
            tokenIdsGetter(ref tokenIdsBuffer);
            attMaskGetter(ref attMaskBuffer);
            tokenIdsList.Add(tokenIdsBuffer.DenseValues().ToArray());
            attentionMaskList.Add(attMaskBuffer.DenseValues().ToArray());

            if (typeIdsGetter != null)
            {
                typeIdsGetter(ref typeIdsBuffer);
                tokenTypeIdsList!.Add(typeIdsBuffer.DenseValues().ToArray());
            }
        }

        return (tokenIdsList.ToArray(), attentionMaskList.ToArray(), tokenTypeIdsList?.ToArray());
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) { /* standard */ }
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();
    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _session.Dispose();
}
```

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingEstimator.DiscoverModelMetadata()` | Tensor auto-discovery | `OnnxTextModelScorerEstimator.DiscoverModelMetadata()` |
| `OnnxTextEmbeddingEstimator.FindTensorName()` | Tensor name lookup | `OnnxTextModelScorerEstimator.FindTensorName()` |
| `OnnxTextEmbeddingEstimator.TryFindTensorName()` | Tensor name lookup | `OnnxTextModelScorerEstimator.TryFindTensorName()` |
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 156-189 | ONNX input/output and inference | `OnnxTextModelScorerTransformer.ProcessBatch()` |

## Key Design Decisions

### Raw output is always per-row

The scorer unpacks the batch ONNX output into per-row `float[]` arrays. For unpooled models, each row gets a `float[seqLen × hiddenDim]` array. This is the memory trade-off discussed in [tradeoffs.md](tradeoffs.md).

For pre-pooled models, each row gets `float[hiddenDim]` — identical to the final embedding.

### No pooling inside the scorer

The scorer is task-agnostic. It doesn't know if the downstream consumer wants mean pooling, CLS token, softmax, or argmax. It outputs whatever the ONNX model outputs.

### Metadata is exposed for downstream transforms

`HiddenDim`, `HasPooledOutput`, and `Metadata` are exposed so downstream transforms (like `EmbeddingPoolingTransformer`) can auto-configure themselves. The facade uses this to wire up the pooling transform without requiring the user to specify dimensions manually.

### Batching is preserved

The scorer batches inference calls (default 32 rows per batch). It reads all token arrays from the IDataView first (eager), then processes in batches. This matches the current monolith's batching strategy.

## Acceptance Criteria

1. `OnnxTextModelScorerEstimator` can be created with a valid ONNX model path
2. `Fit()` validates that token columns exist in the input schema
3. `Fit()` auto-discovers ONNX tensor metadata (input/output names, dimensions)
4. `Transform()` reads token columns, runs ONNX inference, outputs raw model tensor
5. Output shape is `float[hiddenDim]` for pre-pooled models, `float[seqLen × hiddenDim]` for unpooled
6. Batching works correctly (processes BatchSize rows per ONNX Run call)
7. `Score()` (direct face) returns the same results without IDataView overhead
8. `Dispose()` disposes the `InferenceSession`
9. Manual tensor name overrides work for non-standard models
