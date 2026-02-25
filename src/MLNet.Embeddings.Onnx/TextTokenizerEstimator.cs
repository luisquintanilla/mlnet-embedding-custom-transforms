using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tokenizers;

namespace MLNet.Embeddings.Onnx;

/// <summary>
/// Configuration for the text tokenizer transform.
/// </summary>
public class TextTokenizerOptions
{
    /// <summary>
    /// Path to the tokenizer vocabulary file.
    /// Supports: vocab.txt (BERT/WordPiece).
    /// </summary>
    public required string TokenizerPath { get; set; }

    /// <summary>Name of the input text column. Default: "Text".</summary>
    public string InputColumnName { get; set; } = "Text";

    /// <summary>Name of the output token IDs column. Default: "TokenIds".</summary>
    public string TokenIdsColumnName { get; set; } = "TokenIds";

    /// <summary>Name of the output attention mask column. Default: "AttentionMask".</summary>
    public string AttentionMaskColumnName { get; set; } = "AttentionMask";

    /// <summary>Name of the output token type IDs column. Default: "TokenTypeIds".</summary>
    public string TokenTypeIdsColumnName { get; set; } = "TokenTypeIds";

    /// <summary>
    /// Maximum number of tokens per input text.
    /// Texts are truncated to this length; shorter texts are zero-padded.
    /// Default: 128.
    /// </summary>
    public int MaxTokenLength { get; set; } = 128;

    /// <summary>
    /// Whether to output the token type IDs column.
    /// Set to false for models that don't use segment embeddings.
    /// Default: true.
    /// </summary>
    public bool OutputTokenTypeIds { get; set; } = true;
}

/// <summary>
/// ML.NET IEstimator that creates a TextTokenizerTransformer.
/// Trivial estimator — nothing to learn from training data.
/// Fit() validates the input schema and loads the tokenizer.
/// </summary>
public sealed class TextTokenizerEstimator : IEstimator<TextTokenizerTransformer>
{
    private readonly MLContext _mlContext;
    private readonly TextTokenizerOptions _options;

    public TextTokenizerEstimator(MLContext mlContext, TextTokenizerOptions options)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!File.Exists(options.TokenizerPath))
            throw new FileNotFoundException($"Tokenizer file not found: {options.TokenizerPath}");
    }

    public TextTokenizerTransformer Fit(IDataView input)
    {
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        var tokenizer = LoadTokenizer(_options.TokenizerPath);
        return new TextTokenizerTransformer(_mlContext, _options, tokenizer);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var inputCol = inputSchema.FirstOrDefault(c => c.Name == _options.InputColumnName);
        if (inputCol.Name == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        if (inputCol.ItemType != TextDataViewType.Instance)
            throw new ArgumentException(
                $"Column '{_options.InputColumnName}' must be of type Text.");

        var result = inputSchema.ToDictionary(x => x.Name);

        AddVectorColumn(result, _options.TokenIdsColumnName, NumberDataViewType.Int64);
        AddVectorColumn(result, _options.AttentionMaskColumnName, NumberDataViewType.Int64);
        if (_options.OutputTokenTypeIds)
            AddVectorColumn(result, _options.TokenTypeIdsColumnName, NumberDataViewType.Int64);

        return new SchemaShape(result.Values);
    }

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

    private static void AddVectorColumn(
        Dictionary<string, SchemaShape.Column> schema,
        string name,
        DataViewType itemType)
    {
        var colCtor = typeof(SchemaShape.Column).GetConstructors(
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)[0];
        var col = (SchemaShape.Column)colCtor.Invoke([
            name,
            SchemaShape.Column.VectorKind.Vector,
            itemType,
            false,
            (SchemaShape?)null
        ]);
        schema[name] = col;
    }
}
