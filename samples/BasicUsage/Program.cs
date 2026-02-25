using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

var modelPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx");
var vocabPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "vocab.txt");

// Resolve paths
modelPath = Path.GetFullPath(modelPath);
vocabPath = Path.GetFullPath(vocabPath);

Console.WriteLine("=== ONNX Text Embedding Transform for ML.NET ===\n");

// --- 1. ML.NET Pipeline Usage ---
Console.WriteLine("1. ML.NET Pipeline Usage");
Console.WriteLine(new string('-', 40));

var mlContext = new MLContext();

var options = new OnnxTextEmbeddingOptions
{
    ModelPath = modelPath,
    TokenizerPath = vocabPath,
    InputColumnName = "Text",
    OutputColumnName = "Embedding",
    MaxTokenLength = 128,
    Pooling = PoolingStrategy.MeanPooling,
    Normalize = true,
    BatchSize = 8
};

var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);

// Create sample data
var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework for .NET" },
    new TextData { Text = "How to cook pasta" },
    new TextData { Text = "Deep learning and neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);

// Fit (trivial — validates model, creates transformer)
Console.WriteLine("Fitting estimator (loading ONNX model + tokenizer)...");
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Embedding dimension: {transformer.EmbeddingDimension}");

// Transform
Console.WriteLine("Generating embeddings...");
var transformed = transformer.Transform(dataView);

// Read results
var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(transformed, reuseRowObject: false).ToList();

foreach (var (item, idx) in embeddings.Select((e, i) => (e, i)))
{
    Console.WriteLine($"  [{idx}] \"{sampleData[idx].Text}\"");
    Console.WriteLine($"       dims={item.Embedding.Length}, first 5: [{string.Join(", ", item.Embedding.Take(5).Select(f => f.ToString("F4")))}]");
}

// --- 2. Cosine Similarity ---
Console.WriteLine($"\n2. Cosine Similarity");
Console.WriteLine(new string('-', 40));

for (int i = 0; i < embeddings.Count; i++)
{
    for (int j = i + 1; j < embeddings.Count; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
        Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
    }
}

// --- 3. Save/Load Round-Trip ---
Console.WriteLine($"\n3. Save/Load Round-Trip");
Console.WriteLine(new string('-', 40));

var savePath = Path.Combine(Path.GetTempPath(), "embedding-model.mlnet");
Console.WriteLine($"  Saving to: {savePath}");
transformer.Save(savePath);
Console.WriteLine($"  File size: {new FileInfo(savePath).Length / 1024 / 1024} MB");

Console.WriteLine("  Loading from saved file...");
using var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, savePath);
Console.WriteLine($"  Loaded embedding dimension: {loaded.EmbeddingDimension}");

// Verify loaded model produces same results
var loadedTransformed = loaded.Transform(dataView);
var loadedEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(loadedTransformed, reuseRowObject: false).ToList();

float maxDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
{
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
    {
        float diff = MathF.Abs(embeddings[i].Embedding[d] - loadedEmbeddings[i].Embedding[d]);
        maxDiff = MathF.Max(maxDiff, diff);
    }
}
Console.WriteLine($"  Max difference after round-trip: {maxDiff:E2} (should be ~0)");

// Clean up
File.Delete(savePath);

// --- 4. MEAI IEmbeddingGenerator Usage ---
Console.WriteLine($"\n4. MEAI IEmbeddingGenerator Usage");
Console.WriteLine(new string('-', 40));

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);

Console.WriteLine($"  Provider: {(generator as OnnxEmbeddingGenerator)?.Metadata.ProviderName}");
Console.WriteLine($"  Model: {(generator as OnnxEmbeddingGenerator)?.Metadata.DefaultModelId}");

var meaiTexts = new[] { "What is .NET?", "Tell me about the .NET framework", "How to cook pasta" };
var meaiEmbeddings = await generator.GenerateAsync(meaiTexts);

Console.WriteLine($"  Generated {meaiEmbeddings.Count} embeddings");
Console.WriteLine($"  Vector dimensions: {meaiEmbeddings[0].Vector.Length}");

float sim01 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[1].Vector.Span);
float sim02 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[2].Vector.Span);
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[1]}\": {sim01:F4}");
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[2]}\": {sim02:F4}");
Console.WriteLine("  (.NET topics should be more similar to each other than to cooking)");

Console.WriteLine("\nDone!");

// Cleanup
transformer.Dispose();

// --- Domain types ---
public class TextData
{
    public string Text { get; set; } = "";
}

public class EmbeddingResult
{
    public string Text { get; set; } = "";
    public float[] Embedding { get; set; } = [];
}
