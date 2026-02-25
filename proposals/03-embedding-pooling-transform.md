# Transform 3: EmbeddingPoolingEstimator / EmbeddingPoolingTransformer

## Purpose

Applies pooling and normalization to raw model output to produce a fixed-length embedding vector. This is the **task-specific post-processing step** for embedding generation. It is one of potentially many post-processing transforms that can sit downstream of `OnnxTextModelScorerTransformer`.

## Why a Separate Transform?

1. **Swappability**: Users can swap pooling strategy without re-running inference. Re-inference is expensive (~10ms per batch); re-pooling is ~0.01ms.
2. **Composability**: Mean pooling, CLS pooling, and max pooling produce different embedding spaces. Being able to switch at the pipeline level is valuable for experimentation.
3. **Separation of concerns**: L2 normalization is a well-defined operation that ML.NET already has (`LpNormNormalizingEstimator`). Our transform bundles it as a convenience, but users could use ML.NET's built-in normalizer instead.
4. **Future post-processing**: Other post-processing transforms (Matryoshka truncation, binary quantization, PCA reduction) can be added alongside this one.

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.Embeddings.Onnx/EmbeddingPoolingEstimator.cs` | Estimator + options class |
| `src/MLNet.Embeddings.Onnx/EmbeddingPoolingTransformer.cs` | Transformer |

## Options Class

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// Configuration for the embedding pooling transform.
/// Reduces raw model output to a fixed-length embedding vector.
/// </summary>
public class EmbeddingPoolingOptions
{
    /// <summary>
    /// Name of the input column containing raw model output.
    /// For unpooled models: VBuffer&lt;float&gt; of length seqLen × hiddenDim.
    /// For pre-pooled models: VBuffer&lt;float&gt; of length hiddenDim.
    /// Default: "RawOutput".
    /// </summary>
    public string InputColumnName { get; set; } = "RawOutput";

    /// <summary>
    /// Name of the attention mask column. Required for mean and max pooling
    /// (to exclude padding tokens). Not needed for CLS pooling or pre-pooled input.
    /// Default: "AttentionMask".
    /// </summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output embedding column. Default: "Embedding".</summary>
    public string OutputColumnName { get; set; } = "Embedding";

    /// <summary>
    /// Pooling strategy for reducing per-token outputs to a single vector.
    /// Ignored when IsPrePooled is true.
    /// Default: MeanPooling.
    /// </summary>
    public PoolingStrategy Pooling { get; set; } = PoolingStrategy.MeanPooling;

    /// <summary>Whether to L2-normalize the output embeddings. Default: true.</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>
    /// Hidden dimension of the model output.
    /// When used via the facade, this is auto-set from scorer metadata.
    /// When used standalone, must be specified by the user.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Sequence length of the unpooled model output.
    /// Only needed for unpooled models. When used via the facade, auto-set from scorer.
    /// </summary>
    public int SequenceLength { get; set; }

    /// <summary>
    /// Whether the input is already pooled (e.g., sentence_embedding output).
    /// When true, only normalization is applied (pooling strategy is ignored).
    /// When used via the facade, auto-set from scorer metadata.
    /// Default: false.
    /// </summary>
    public bool IsPrePooled { get; set; }
}
```

## Estimator

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an EmbeddingPoolingTransformer.
/// Trivial estimator — validates schema and passes configuration through.
/// </summary>
public sealed class EmbeddingPoolingEstimator : IEstimator<EmbeddingPoolingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly EmbeddingPoolingOptions _options;

    public EmbeddingPoolingEstimator(MLContext mlContext, EmbeddingPoolingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (options.HiddenDim <= 0)
            throw new ArgumentException("HiddenDim must be positive.", nameof(options));

        if (!options.IsPrePooled && options.SequenceLength <= 0)
            throw new ArgumentException(
                "SequenceLength must be positive for unpooled models.", nameof(options));
    }

    /// <summary>
    /// Creates an EmbeddingPoolingEstimator that auto-configures from scorer metadata.
    /// Used internally by the facade.
    /// </summary>
    internal EmbeddingPoolingEstimator(
        MLContext mlContext,
        EmbeddingPoolingOptions options,
        OnnxTextModelScorerTransformer scorer)
        : this(mlContext, ConfigureFromScorer(options, scorer))
    {
    }

    private static EmbeddingPoolingOptions ConfigureFromScorer(
        EmbeddingPoolingOptions options,
        OnnxTextModelScorerTransformer scorer)
    {
        // Auto-fill dimensions from scorer metadata
        options.HiddenDim = scorer.HiddenDim;
        options.IsPrePooled = scorer.HasPooledOutput;
        if (!scorer.HasPooledOutput)
            options.SequenceLength = scorer.Options.MaxTokenLength;
        return options;
    }

    public EmbeddingPoolingTransformer Fit(IDataView input)
    {
        // Validate input schema has the raw output column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Validate attention mask column exists (if needed for pooling)
        if (!_options.IsPrePooled && _options.Pooling != PoolingStrategy.ClsToken)
        {
            var maskCol = input.Schema.GetColumnOrNull(_options.AttentionMaskColumnName);
            if (maskCol == null)
                throw new ArgumentException(
                    $"Input schema does not contain column '{_options.AttentionMaskColumnName}'. " +
                    $"Required for {_options.Pooling} pooling.");
        }

        return new EmbeddingPoolingTransformer(_mlContext, _options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input column exists
        // Add output column: Vector<float> of size HiddenDim
        var result = inputSchema.ToDictionary(x => x.Name);

        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var outputCol = (SchemaShape.Column)colCtor.Invoke([
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            (DataViewType)NumberDataViewType.Single,
            false,
            (SchemaShape?)null
        ]);
        result[_options.OutputColumnName] = outputCol;

        return new SchemaShape(result.Values);
    }
}
```

## Transformer

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET ITransformer that pools raw model output into fixed-length embeddings.
/// Supports mean, CLS, and max pooling, plus optional L2 normalization.
/// </summary>
public sealed class EmbeddingPoolingTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly EmbeddingPoolingOptions _options;

    public bool IsRowToRowMapper => true;

    internal EmbeddingPoolingOptions Options => _options;
    public int EmbeddingDimension => _options.HiddenDim;

    internal EmbeddingPoolingTransformer(
        MLContext mlContext,
        EmbeddingPoolingOptions options)
    {
        _mlContext = mlContext;
        _options = options;
    }

    public IDataView Transform(IDataView input)
    {
        // 1. Read raw output and attention mask columns
        var (rawOutputs, attentionMasks, texts) = ReadColumns(input);

        // 2. Apply pooling
        var embeddings = Pool(rawOutputs, attentionMasks);

        // 3. Build output IDataView
        return BuildOutputDataView(texts, embeddings);
    }

    /// <summary>
    /// Direct face: pool raw outputs without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal float[][] Pool(float[][] rawOutputs, long[][]? attentionMasks)
    {
        if (_options.IsPrePooled)
        {
            // Pre-pooled: just normalize if requested
            if (_options.Normalize)
            {
                for (int i = 0; i < rawOutputs.Length; i++)
                    L2Normalize(rawOutputs[i]);
            }
            return rawOutputs;
        }

        // Unpooled: apply pooling strategy
        int hiddenDim = _options.HiddenDim;
        int seqLen = _options.SequenceLength;
        var embeddings = new float[rawOutputs.Length][];

        for (int i = 0; i < rawOutputs.Length; i++)
        {
            // rawOutputs[i] is flat [seqLen × hiddenDim]
            // attentionMasks[i] is [seqLen]
            ReadOnlySpan<float> hiddenStates = rawOutputs[i];
            ReadOnlySpan<long> mask = attentionMasks![i];

            embeddings[i] = _options.Pooling switch
            {
                // Delegate to existing EmbeddingPooling static methods.
                // These currently take batch parameters, so we pass batchSize=1, batchIdx=0.
                PoolingStrategy.MeanPooling =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.MeanPooling, false)[0],
                PoolingStrategy.ClsToken =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.ClsToken, false)[0],
                PoolingStrategy.MaxPooling =>
                    EmbeddingPooling.Pool(hiddenStates, mask, 1, seqLen, hiddenDim,
                        PoolingStrategy.MaxPooling, false)[0],
                _ => throw new ArgumentOutOfRangeException()
            };

            if (_options.Normalize)
                L2Normalize(embeddings[i]);
        }

        return embeddings;
    }

    private static void L2Normalize(Span<float> embedding)
    {
        float norm = TensorPrimitives.Norm(embedding);
        if (norm > 0)
            TensorPrimitives.Divide(embedding, norm, embedding);
    }

    private (float[][] rawOutputs, long[][]? attentionMasks, List<string> texts) ReadColumns(IDataView input)
    {
        // Read RawOutput (VBuffer<float>) and AttentionMask (VBuffer<long>) via cursor
        // Also read Text column for passthrough
        // ...standard cursor pattern...
    }

    private IDataView BuildOutputDataView(List<string> texts, float[][] embeddings)
    {
        var rows = new List<EmbeddingRow>(texts.Count);
        for (int i = 0; i < texts.Count; i++)
        {
            rows.Add(new EmbeddingRow
            {
                Text = texts[i],
                Embedding = embeddings[i]
            });
        }
        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) { /* standard */ }
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();
    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    private sealed class EmbeddingRow
    {
        public string Text { get; set; } = "";

        [VectorType]
        public float[] Embedding { get; set; } = [];
    }
}
```

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 173-183 | Pooling dispatch logic | `EmbeddingPoolingTransformer.Pool()` |
| `EmbeddingPooling.cs` | Unchanged — continues to provide the static math | Used by `EmbeddingPoolingTransformer` |

## Relationship to EmbeddingPooling.cs

`EmbeddingPooling.cs` (the existing static class) is **not modified**. It provides the SIMD-accelerated math (`MeanPool`, `ClsPool`, `MaxPool`, `L2Normalize`). The new `EmbeddingPoolingTransformer` wraps it with IDataView plumbing and the direct face API.

The `EmbeddingPooling` static class methods currently take batch-oriented parameters (`batchSize`, `batchIdx`). When called from the transformer's per-row `Pool()` method, we pass `batchSize=1, batchIdx=0`. This works correctly but is slightly wasteful (the batch loop runs once). A minor optimization would be to add per-row convenience methods to `EmbeddingPooling`, but it's not necessary for correctness.

## Future Post-Processing Transforms

The embedding pooling transform is the first in what could be a family of post-processing transforms:

| Transform | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **EmbeddingPoolingTransformer** (this) | Mean/CLS/Max + normalize | Raw hidden states | `float[hiddenDim]` |
| **MatryoshkaTruncationTransformer** | Truncate to N dims | `float[hiddenDim]` | `float[N]` where N < hiddenDim |
| **BinaryQuantizationTransformer** | 1-bit per dim | `float[hiddenDim]` | `byte[hiddenDim/8]` |
| **SoftmaxClassificationTransformer** | Logits → probabilities | `float[numClasses]` | `float[numClasses]` |
| **NerDecodingTransformer** | Per-token logits → entities | `float[seqLen × numLabels]` | Entity spans |

All share the same pattern: read a VBuffer<float> column, apply math, write a new column.

## Acceptance Criteria

1. `EmbeddingPoolingEstimator` validates HiddenDim and SequenceLength
2. Auto-configures from `OnnxTextModelScorerTransformer` metadata when used via facade
3. `Fit()` validates that input and attention mask columns exist
4. `Transform()` applies mean, CLS, or max pooling correctly
5. Pre-pooled pass-through works (only normalizes)
6. L2 normalization is optional and works correctly
7. `Pool()` (direct face) returns the same results without IDataView overhead
8. Embedding dimension is exposed via `EmbeddingDimension` property
