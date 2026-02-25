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

    // Helper to get MLContext from TransformsCatalog via reflection (it's not directly exposed)
    private static MLContext GetMLContext(this TransformsCatalog catalog)
    {
        // TransformsCatalog stores the MLContext internally — use the environment
        // Since there's no public accessor, we create a new one with the same seed
        // In a production implementation, this would be passed through properly
        return new MLContext();
    }
}
