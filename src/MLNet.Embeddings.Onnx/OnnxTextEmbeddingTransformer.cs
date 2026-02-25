using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Runtime;
using Microsoft.ML.Tokenizers;

namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET ITransformer that generates text embeddings using a local ONNX model.
/// Encapsulates tokenization → ONNX inference → pooling in a single transform.
/// </summary>
public sealed class OnnxTextEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;
    private readonly InferenceSession _session;
    private readonly Tokenizer _tokenizer;

    // Discovered ONNX metadata
    private readonly string _inputIdsName;
    private readonly string _attentionMaskName;
    private readonly string? _tokenTypeIdsName;
    private readonly string _outputTensorName;
    private readonly int _hiddenDim;
    private readonly bool _modelHasPooledOutput;

    public bool IsRowToRowMapper => true;

    internal OnnxTextEmbeddingOptions Options => _options;
    public int EmbeddingDimension => _hiddenDim;

    internal OnnxTextEmbeddingTransformer(
        MLContext mlContext,
        OnnxTextEmbeddingOptions options,
        InferenceSession session,
        Tokenizer tokenizer,
        string inputIdsName,
        string attentionMaskName,
        string? tokenTypeIdsName,
        string outputTensorName,
        int hiddenDim,
        bool modelHasPooledOutput)
    {
        _mlContext = mlContext;
        _options = options;
        _session = session;
        _tokenizer = tokenizer;
        _inputIdsName = inputIdsName;
        _attentionMaskName = attentionMaskName;
        _tokenTypeIdsName = tokenTypeIdsName;
        _outputTensorName = outputTensorName;
        _hiddenDim = hiddenDim;
        _modelHasPooledOutput = modelHasPooledOutput;
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumns(inputSchema);

        // Add embedding column
        var embeddingType = new VectorDataViewType(NumberDataViewType.Single, _hiddenDim);
        builder.AddColumn(_options.OutputColumnName, embeddingType);

        return builder.ToSchema();
    }

    public IDataView Transform(IDataView input)
    {
        // 1. Read all text from input column
        var texts = ReadTextColumn(input);
        if (texts.Count == 0)
            return CreateEmptyOutput(input);

        // 2. Process in batches
        var allEmbeddings = new List<float[]>(texts.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, texts.Count - start);
            var batchTexts = texts.GetRange(start, count);
            var batchEmbeddings = ProcessBatch(batchTexts);
            allEmbeddings.AddRange(batchEmbeddings);
        }

        // 3. Build output IDataView with original columns + embedding column
        return BuildOutputDataView(input, allEmbeddings);
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        throw new NotSupportedException(
            "Row-to-row mapping is not supported in this prototype. " +
            "Use Transform() for batch processing.");
    }

    void ICanSaveModel.Save(ModelSaveContext ctx)
    {
        throw new NotSupportedException(
            "ML.NET native save is not supported. Use transformer.Save(path) instead.");
    }

    /// <summary>
    /// Generates embeddings for a list of texts directly (bypasses IDataView).
    /// Useful for the MEAI wrapper.
    /// </summary>
    internal float[][] GenerateEmbeddings(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0)
            return [];

        var allEmbeddings = new List<float[]>(texts.Count);
        int batchSize = _options.BatchSize;

        for (int start = 0; start < texts.Count; start += batchSize)
        {
            int count = Math.Min(batchSize, texts.Count - start);
            var batchTexts = new List<string>(count);
            for (int i = start; i < start + count; i++)
                batchTexts.Add(texts[i]);

            var batchEmbeddings = ProcessBatch(batchTexts);
            allEmbeddings.AddRange(batchEmbeddings);
        }

        return [.. allEmbeddings];
    }

    private float[][] ProcessBatch(List<string> texts)
    {
        int batchSize = texts.Count;
        int seqLen = _options.MaxTokenLength;

        // Tokenize and build input tensors using Tensor<T> for shape-safe indexing
        var idsArray = new long[batchSize * seqLen];
        var maskArray = new long[batchSize * seqLen];
        var typeIdsArray = _tokenTypeIdsName != null ? new long[batchSize * seqLen] : null;

        var idsTensor = Tensor.Create<long>(idsArray, [batchSize, seqLen]);
        var maskTensor = Tensor.Create<long>(maskArray, [batchSize, seqLen]);

        for (int b = 0; b < batchSize; b++)
        {
            var tokens = _tokenizer.EncodeToIds(texts[b], seqLen, out _, out _);

            for (int s = 0; s < tokens.Count && s < seqLen; s++)
            {
                idsTensor[b, s] = tokens[s];
                maskTensor[b, s] = 1;
            }
        }

        // Create OrtValues from flat backing arrays (zero-copy)
        var inputs = new Dictionary<string, OrtValue>
        {
            [_inputIdsName] = OrtValue.CreateTensorValueFromMemory(idsArray, [batchSize, seqLen]),
            [_attentionMaskName] = OrtValue.CreateTensorValueFromMemory(maskArray, [batchSize, seqLen])
        };

        if (_tokenTypeIdsName != null && typeIdsArray != null)
            inputs[_tokenTypeIdsName] = OrtValue.CreateTensorValueFromMemory(typeIdsArray, [batchSize, seqLen]);

        try
        {
            // Run ONNX inference
            using var results = _session.Run(new RunOptions(), inputs, [_outputTensorName]);
            var output = results[0];
            var outputSpan = output.GetTensorDataAsSpan<float>();

            // Pool and normalize
            if (_modelHasPooledOutput)
            {
                return EmbeddingPooling.ExtractPooled(outputSpan, batchSize, _hiddenDim, _options.Normalize);
            }
            else
            {
                return EmbeddingPooling.Pool(
                    outputSpan, maskArray, batchSize, seqLen, _hiddenDim,
                    _options.Pooling, _options.Normalize);
            }
        }
        finally
        {
            foreach (var ortValue in inputs.Values)
                ortValue.Dispose();
        }
    }

    private List<string> ReadTextColumn(IDataView dataView)
    {
        var texts = new List<string>();
        var col = dataView.Schema[_options.InputColumnName];
        using var cursor = dataView.GetRowCursor(new[] { col });
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(col);

        ReadOnlyMemory<char> value = default;
        while (cursor.MoveNext())
        {
            getter(ref value);
            texts.Add(value.ToString());
        }

        return texts;
    }

    private IDataView CreateEmptyOutput(IDataView input)
    {
        return BuildOutputDataView(input, []);
    }

    private IDataView BuildOutputDataView(IDataView input, List<float[]> embeddings)
    {
        // Read all input rows + attach embeddings
        var inputSchema = input.Schema;
        var rows = new List<EmbeddingRow>();

        var textCol = inputSchema[_options.InputColumnName];
        using var cursor = input.GetRowCursor(new[] { textCol });
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(textCol);

        int idx = 0;
        ReadOnlyMemory<char> value = default;
        while (cursor.MoveNext())
        {
            getter(ref value);
            rows.Add(new EmbeddingRow
            {
                Text = value.ToString(),
                Embedding = idx < embeddings.Count ? embeddings[idx] : []
            });
            idx++;
        }

        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    /// <summary>
    /// Saves the transformer to a self-contained zip file.
    /// </summary>
    public void Save(string path) => ModelPackager.Save(this, path);

    /// <summary>
    /// Loads a transformer from a saved zip file.
    /// </summary>
    public static OnnxTextEmbeddingTransformer Load(MLContext mlContext, string path)
        => ModelPackager.Load(mlContext, path);

    public void Dispose()
    {
        _session.Dispose();
    }

    // Internal POCO for building output IDataView
    private sealed class EmbeddingRow
    {
        public string Text { get; set; } = "";

        [VectorType]
        public float[] Embedding { get; set; } = [];
    }
}
