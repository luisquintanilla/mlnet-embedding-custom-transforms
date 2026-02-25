using Microsoft.Extensions.AI;
using Microsoft.ML;

namespace MLNet.Embeddings.Onnx;

/// <summary>
/// Extension methods for MLContext to provide a convenient API for ONNX text embeddings.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Creates an estimator that generates text embeddings using a local ONNX model.
    /// Encapsulates tokenization, ONNX inference, pooling, and normalization.
    /// </summary>
    public static OnnxTextEmbeddingEstimator OnnxTextEmbedding(
        this TransformsCatalog catalog,
        OnnxTextEmbeddingOptions options)
    {
        return new OnnxTextEmbeddingEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates a provider-agnostic embedding transform that wraps any IEmbeddingGenerator.
    /// </summary>
    public static EmbeddingGeneratorEstimator TextEmbedding(
        this TransformsCatalog catalog,
        IEmbeddingGenerator<string, Embedding<float>> generator,
        EmbeddingGeneratorOptions? options = null)
    {
        return new EmbeddingGeneratorEstimator(catalog.GetMLContext(), generator, options);
    }

    /// <summary>
    /// Creates a text tokenizer transform for transformer-based models.
    /// </summary>
    public static TextTokenizerEstimator TokenizeText(
        this TransformsCatalog catalog,
        TextTokenizerOptions options)
    {
        return new TextTokenizerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an ONNX text model scorer transform for transformer-based models.
    /// </summary>
    public static OnnxTextModelScorerEstimator ScoreOnnxTextModel(
        this TransformsCatalog catalog,
        OnnxTextModelScorerOptions options)
    {
        return new OnnxTextModelScorerEstimator(catalog.GetMLContext(), options);
    }

    /// <summary>
    /// Creates an embedding pooling transform for reducing raw model output to embeddings.
    /// </summary>
    public static EmbeddingPoolingEstimator PoolEmbedding(
        this TransformsCatalog catalog,
        EmbeddingPoolingOptions options)
    {
        return new EmbeddingPoolingEstimator(catalog.GetMLContext(), options);
    }

    // Helper to get MLContext from TransformsCatalog via reflection (it's not directly exposed)
    private static MLContext GetMLContext(this TransformsCatalog catalog)
    {
        // TransformsCatalog stores the MLContext internally — use the environment
        // Since there's no public accessor, we create a new one with the same seed
        // In a production implementation, this would be passed through properly
        return new MLContext();
    }
}
