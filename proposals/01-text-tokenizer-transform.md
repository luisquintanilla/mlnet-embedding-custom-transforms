# Transform 1: TextTokenizerEstimator / TextTokenizerTransformer

## Purpose

Tokenizes text into token IDs, attention masks, and token type IDs suitable for any transformer-based ONNX model. This is the **universal first step** for any transformer task (embeddings, classification, NER, QA, etc.).

## Files to Create

| File | Contents |
|------|----------|
| `src/MLNet.Embeddings.Onnx/TextTokenizerEstimator.cs` | Estimator + options class |
| `src/MLNet.Embeddings.Onnx/TextTokenizerTransformer.cs` | Transformer |

## Options Class

```csharp
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
```

## Estimator

```csharp
namespace MLNet.Embeddings.Onnx;

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
        // Validate input schema has the text column
        var col = input.Schema.GetColumnOrNull(_options.InputColumnName);
        if (col == null)
            throw new ArgumentException(
                $"Input schema does not contain column '{_options.InputColumnName}'.");

        // Load tokenizer (reuse existing LoadTokenizer logic)
        var tokenizer = LoadTokenizer(_options.TokenizerPath);

        return new TextTokenizerTransformer(_mlContext, _options, tokenizer);
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
                $"Column '{_options.InputColumnName}' must be of type Text.");

        // Build output schema: input columns + token columns
        var result = inputSchema.ToDictionary(x => x.Name);

        // Add TokenIds, AttentionMask, TokenTypeIds as Vector<Int64> columns
        // (uses same reflection workaround as current GetOutputSchema)
        AddVectorColumn(result, _options.TokenIdsColumnName, NumberDataViewType.Int64);
        AddVectorColumn(result, _options.AttentionMaskColumnName, NumberDataViewType.Int64);
        if (_options.OutputTokenTypeIds)
            AddVectorColumn(result, _options.TokenTypeIdsColumnName, NumberDataViewType.Int64);

        return new SchemaShape(result.Values);
    }

    // Reuse from existing OnnxTextEmbeddingEstimator.LoadTokenizer()
    internal static Tokenizer LoadTokenizer(string path) { /* same logic */ }

    private static void AddVectorColumn(
        Dictionary<string, SchemaShape.Column> schema,
        string name,
        DataViewType itemType)
    {
        // Same reflection workaround as current GetOutputSchema
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
```

## Transformer

```csharp
namespace MLNet.Embeddings.Onnx;

/// <summary>
/// ML.NET ITransformer that tokenizes text into token IDs, attention masks,
/// and token type IDs. Produces fixed-length padded/truncated output.
/// </summary>
public sealed class TextTokenizerTransformer : ITransformer
{
    private readonly MLContext _mlContext;
    private readonly TextTokenizerOptions _options;
    private readonly Tokenizer _tokenizer;

    public bool IsRowToRowMapper => true;

    internal TextTokenizerOptions Options => _options;

    internal TextTokenizerTransformer(
        MLContext mlContext,
        TextTokenizerOptions options,
        Tokenizer tokenizer)
    {
        _mlContext = mlContext;
        _options = options;
        _tokenizer = tokenizer;
    }

    public IDataView Transform(IDataView input)
    {
        // 1. Read all text from input column
        var texts = ReadTextColumn(input);

        // 2. Tokenize all texts
        var tokenized = Tokenize(texts);

        // 3. Build output IDataView with original text + token columns
        return BuildOutputDataView(texts, tokenized);
    }

    /// <summary>
    /// Direct face: tokenize a list of texts without IDataView overhead.
    /// Used by the facade and MEAI generator.
    /// </summary>
    internal TokenizedBatch Tokenize(IReadOnlyList<string> texts)
    {
        int seqLen = _options.MaxTokenLength;
        var allTokenIds = new long[texts.Count][];
        var allAttentionMasks = new long[texts.Count][];
        var allTokenTypeIds = _options.OutputTokenTypeIds ? new long[texts.Count][] : null;

        for (int i = 0; i < texts.Count; i++)
        {
            var tokenIds = new long[seqLen];
            var attentionMask = new long[seqLen];
            var tokenTypeIds = _options.OutputTokenTypeIds ? new long[seqLen] : null;

            var tokens = _tokenizer.EncodeToIds(texts[i], seqLen, out _, out _);

            for (int s = 0; s < tokens.Count && s < seqLen; s++)
            {
                tokenIds[s] = tokens[s];
                attentionMask[s] = 1;
                // tokenTypeIds stays 0 (single-segment)
            }

            allTokenIds[i] = tokenIds;
            allAttentionMasks[i] = attentionMask;
            if (allTokenTypeIds != null)
                allTokenTypeIds[i] = tokenTypeIds!;
        }

        return new TokenizedBatch(allTokenIds, allAttentionMasks, allTokenTypeIds, seqLen);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) { /* standard */ }
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new NotSupportedException();
    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    private List<string> ReadTextColumn(IDataView dataView)
    {
        // Same pattern as current OnnxTextEmbeddingTransformer.ReadTextColumn()
    }

    private IDataView BuildOutputDataView(
        IReadOnlyList<string> texts,
        TokenizedBatch tokenized)
    {
        var rows = new List<TokenizedRow>(texts.Count);
        for (int i = 0; i < texts.Count; i++)
        {
            rows.Add(new TokenizedRow
            {
                Text = texts[i],
                TokenIds = tokenized.TokenIds[i],
                AttentionMask = tokenized.AttentionMasks[i],
                TokenTypeIds = tokenized.TokenTypeIds?[i] ?? []
            });
        }
        return _mlContext.Data.LoadFromEnumerable(rows);
    }

    private sealed class TokenizedRow
    {
        public string Text { get; set; } = "";

        [VectorType]
        public long[] TokenIds { get; set; } = [];

        [VectorType]
        public long[] AttentionMask { get; set; } = [];

        [VectorType]
        public long[] TokenTypeIds { get; set; } = [];
    }
}

/// <summary>
/// Batch of tokenized text. Used by the direct face to pass data between transforms
/// without IDataView overhead.
/// </summary>
internal sealed class TokenizedBatch
{
    public long[][] TokenIds { get; }
    public long[][] AttentionMasks { get; }
    public long[][]? TokenTypeIds { get; }
    public int SequenceLength { get; }

    public TokenizedBatch(long[][] tokenIds, long[][] attentionMasks, long[][]? tokenTypeIds, int seqLen)
    {
        TokenIds = tokenIds;
        AttentionMasks = attentionMasks;
        TokenTypeIds = tokenTypeIds;
        SequenceLength = seqLen;
    }

    public int Count => TokenIds.Length;
}
```

## Code to Extract From Existing Files

| Source | What to Extract | Target |
|--------|----------------|--------|
| `OnnxTextEmbeddingEstimator.LoadTokenizer()` | Tokenizer loading logic | `TextTokenizerEstimator.LoadTokenizer()` |
| `OnnxTextEmbeddingTransformer.ProcessBatch()` lines 145-154 | Tokenization loop | `TextTokenizerTransformer.Tokenize()` |
| `OnnxTextEmbeddingTransformer.ReadTextColumn()` | Text column reading | `TextTokenizerTransformer.ReadTextColumn()` |

## Column Passthrough Strategy

The tokenizer reads only the `InputColumnName` column but the output IDataView should contain:
- The original text column (passed through)
- TokenIds (new)
- AttentionMask (new)
- TokenTypeIds (new, optional)

Other input columns are NOT passed through. This simplifies the implementation and avoids the complexity of generic column forwarding. The facade handles reconstructing the full output.

## Acceptance Criteria

1. `TextTokenizerEstimator` can be created with a valid tokenizer path
2. `Fit()` validates the input schema has the text column
3. `Transform()` produces an IDataView with TokenIds, AttentionMask, TokenTypeIds columns
4. Token arrays are padded to MaxTokenLength with zeros
5. Attention masks are 1 for real tokens, 0 for padding
6. `Tokenize()` (direct face) returns the same results without IDataView overhead
7. Works with vocab.txt (BertTokenizer)
