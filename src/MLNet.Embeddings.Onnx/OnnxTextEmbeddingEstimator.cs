using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET IEstimator that creates an OnnxTextEmbeddingTransformer.
/// This is a trivial estimator — there's nothing to learn from training data.
/// Fit() validates the ONNX model, auto-discovers tensor metadata, and returns the transformer.
/// </summary>
public sealed class OnnxTextEmbeddingEstimator : IEstimator<OnnxTextEmbeddingTransformer>
{
    private readonly MLContext _mlContext;
    private readonly OnnxTextEmbeddingOptions _options;

    public OnnxTextEmbeddingEstimator(MLContext mlContext, OnnxTextEmbeddingOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.ModelPath))
            throw new FileNotFoundException($"ONNX model not found: {options.ModelPath}");
        if (!File.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer file not found: {options.TokenizerPath}");
    }

    public OnnxTextEmbeddingTransformer Fit(IDataView input)
    {
        // Validate input schema has the text column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Load ONNX model and auto-discover tensor metadata
        var session = new InferenceSession(_options.ModelPath);
        var (inputIdsName, attentionMaskName, tokenTypeIdsName, outputName, hiddenDim, hasPooledOutput) =
            DiscoverModelMetadata(session);

        // Load tokenizer
        var tokenizer = LoadTokenizer(_options.TokenizerPath);

        return new OnnxTextEmbeddingTransformer(
            _mlContext, _options, session, tokenizer,
            inputIdsName, attentionMaskName, tokenTypeIdsName,
            outputName, hiddenDim, hasPooledOutput);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        // Validate input column exists and is text
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        if (inputCol.ItemType != TextDataViewType.Instance)
            throw new ArgumentException(
                $"Column '{_options.InputColumnName}' must be of type Text, but is {inputCol.ItemType}.");

        // Probe the model to get embedding dimension
        int embeddingDim;
        using (var session = new InferenceSession(_options.ModelPath))
        {
            var (_, _, _, _, dim, _) = DiscoverModelMetadata(session);
            embeddingDim = dim;
        }

        var result = inputSchema.ToDictionary(x => x.Name);
        // SchemaShape.Column has a non-public constructor; use reflection to create it
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

    internal (string inputIdsName, string attentionMaskName, string? tokenTypeIdsName,
             string outputName, int hiddenDim, bool hasPooledOutput)
        DiscoverModelMetadata(InferenceSession session)
    {
        var inputMeta = session.InputMetadata;
        var outputMeta = session.OutputMetadata;

        // Discover input tensor names
        string inputIdsName = _options.InputIdsName
            ?? FindTensorName(inputMeta, ["input_ids"], "input_ids");
        string attentionMaskName = _options.AttentionMaskName
            ?? FindTensorName(inputMeta, ["attention_mask"], "attention_mask");
        string? tokenTypeIdsName = _options.TokenTypeIdsName
            ?? TryFindTensorName(inputMeta, ["token_type_ids"]);

        // Discover output tensor name and determine if model has pre-pooled output
        bool hasPooledOutput;
        string outputName;
        int hiddenDim;

        if (_options.OutputTensorName != null)
        {
            outputName = _options.OutputTensorName;
            hasPooledOutput = !outputMeta[outputName].Dimensions.Contains(-1) &&
                              outputMeta[outputName].Dimensions.Length == 2;
            hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
        }
        else
        {
            // Prefer sentence_embedding (pre-pooled) if available
            var pooledName = TryFindTensorName(outputMeta, ["sentence_embedding", "pooler_output"]);
            if (pooledName != null)
            {
                outputName = pooledName;
                hasPooledOutput = true;
                hiddenDim = (int)outputMeta[pooledName].Dimensions.Last();
            }
            else
            {
                // Fall back to last_hidden_state (needs pooling)
                outputName = FindTensorName(outputMeta,
                    ["last_hidden_state", "output", "hidden_states"],
                    outputMeta.Keys.First());
                hasPooledOutput = false;
                hiddenDim = (int)outputMeta[outputName].Dimensions.Last();
            }
        }

        if (hiddenDim <= 0)
            throw new InvalidOperationException(
                $"Could not determine embedding dimension from ONNX output '{outputName}'. " +
                $"Dimensions: [{string.Join(", ", outputMeta[outputName].Dimensions)}]");

        return (inputIdsName, attentionMaskName, tokenTypeIdsName, outputName, hiddenDim, hasPooledOutput);
    }

    private static string FindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string[] candidates,
        string fallback)
    {
        return TryFindTensorName(metadata, candidates) ?? fallback;
    }

    private static string? TryFindTensorName(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (metadata.ContainsKey(candidate))
                return candidate;
        }
        return null;
    }

    /// <summary>
    /// Loads a tokenizer from a vocab file. Supports:
    /// - vocab.txt (WordPiece/BERT tokenizer, used by MiniLM, BERT, etc.)
    /// - vocab.json (BPE tokenizer, used by GPT-2, etc.)
    /// </summary>
    internal static Tokenizer LoadTokenizer(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        using var stream = File.OpenRead(path);

        return ext switch
        {
            ".txt" => BertTokenizer.Create(stream),
            _ => throw new NotSupportedException(
                $"Unsupported tokenizer file format '{ext}'. " +
                $"Use vocab.txt for BERT/WordPiece models.")
        };
    }
}
