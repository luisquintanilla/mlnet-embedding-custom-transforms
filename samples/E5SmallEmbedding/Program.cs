using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var vocabPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "vocab.txt"));

Console.WriteLine("=== E5-Small-v2 Embedding Sample ===\n");

var mlContext = new MLContext();

// --- 1. Standard embedding (no prefix) ---
Console.WriteLine("1. Standard Embedding (no prefix)");
Console.WriteLine(new string('-', 40));

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

var sampleData = new[]
{
    new TextData { Text = "What is machine learning?" },
    new TextData { Text = "ML.NET is a machine learning framework" },
    new TextData { Text = "How to bake sourdough bread" },
    new TextData { Text = "Deep learning uses neural networks" }
};

var dataView = mlContext.Data.LoadFromEnumerable(sampleData);
var transformer = estimator.Fit(dataView);
Console.WriteLine($"  Embedding dimension: {transformer.EmbeddingDimension}");

var transformed = transformer.Transform(dataView);
var embeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(transformed, reuseRowObject: false).ToList();

for (int i = 0; i < embeddings.Count; i++)
    for (int j = i + 1; j < embeddings.Count; j++)
    {
        float sim = TensorPrimitives.CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
        Console.WriteLine($"  \"{sampleData[i].Text}\" vs \"{sampleData[j].Text}\": {sim:F4}");
    }

// --- 2. Retrieval with query/passage prefixes ---
Console.WriteLine("\n2. Retrieval with E5 Prefixes");
Console.WriteLine(new string('-', 40));
Console.WriteLine("  E5 uses 'query: ' for queries and 'passage: ' for documents.");

// Embed passages WITH "passage: " prefix
var passages = new[]
{
    new TextData { Text = "passage: Machine learning is a subset of artificial intelligence." },
    new TextData { Text = "passage: Bread baking requires flour, water, yeast, and salt." },
    new TextData { Text = "passage: Neural networks are inspired by biological neurons." }
};

var passageView = mlContext.Data.LoadFromEnumerable(passages);
var passageTransformed = transformer.Transform(passageView);
var passageEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(passageTransformed, reuseRowObject: false).ToList();

// Original passage texts for display
var passageTexts = new[]
{
    "Machine learning is a subset of artificial intelligence.",
    "Bread baking requires flour, water, yeast, and salt.",
    "Neural networks are inspired by biological neurons."
};

// Query WITHOUT prefix
Console.WriteLine("  Without 'query: ' prefix:");
var queryNoPrefix = new[] { new TextData { Text = "What is AI?" } };
var queryNoPrefixView = mlContext.Data.LoadFromEnumerable(queryNoPrefix);
var queryNoPrefixEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(
    transformer.Transform(queryNoPrefixView), reuseRowObject: false).ToList();

for (int i = 0; i < passageTexts.Length; i++)
{
    float sim = TensorPrimitives.CosineSimilarity(queryNoPrefixEmbeddings[0].Embedding, passageEmbeddings[i].Embedding);
    Console.WriteLine($"    vs \"{passageTexts[i]}\": {sim:F4}");
}

// Query WITH "query: " prefix
Console.WriteLine("  With 'query: ' prefix:");
var queryWithPrefix = new[] { new TextData { Text = "query: What is AI?" } };
var queryWithPrefixView = mlContext.Data.LoadFromEnumerable(queryWithPrefix);
var queryWithPrefixEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(
    transformer.Transform(queryWithPrefixView), reuseRowObject: false).ToList();

for (int i = 0; i < passageTexts.Length; i++)
{
    float sim = TensorPrimitives.CosineSimilarity(queryWithPrefixEmbeddings[0].Embedding, passageEmbeddings[i].Embedding);
    Console.WriteLine($"    vs \"{passageTexts[i]}\": {sim:F4}");
}

// --- 3. Save/Load Round-Trip ---
Console.WriteLine("\n3. Save/Load Round-Trip");
Console.WriteLine(new string('-', 40));

var savePath = Path.Combine(Path.GetTempPath(), "e5-small-embedding.mlnet");
Console.WriteLine($"  Saving to: {savePath}");
transformer.Save(savePath);
Console.WriteLine($"  File size: {new FileInfo(savePath).Length / 1024 / 1024} MB");

Console.WriteLine("  Loading from saved file...");
using var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, savePath);
Console.WriteLine($"  Loaded embedding dimension: {loaded.EmbeddingDimension}");

var loadedTransformed = loaded.Transform(dataView);
var loadedEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(loadedTransformed, reuseRowObject: false).ToList();

float maxDiff = 0;
for (int i = 0; i < embeddings.Count; i++)
    for (int d = 0; d < embeddings[i].Embedding.Length; d++)
        maxDiff = MathF.Max(maxDiff, MathF.Abs(embeddings[i].Embedding[d] - loadedEmbeddings[i].Embedding[d]));
Console.WriteLine($"  Max difference after round-trip: {maxDiff:E2} (should be ~0)");
File.Delete(savePath);

// --- 4. MEAI IEmbeddingGenerator ---
Console.WriteLine($"\n4. MEAI IEmbeddingGenerator Usage");
Console.WriteLine(new string('-', 40));

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, transformer);

var meaiTexts = new[] { "What is .NET?", "Tell me about the .NET framework", "How to cook pasta" };
var meaiEmbeddings = await generator.GenerateAsync(meaiTexts);

Console.WriteLine($"  Generated {meaiEmbeddings.Count} embeddings");
Console.WriteLine($"  Vector dimensions: {meaiEmbeddings[0].Vector.Length}");

float sim01 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[1].Vector.Span);
float sim02 = TensorPrimitives.CosineSimilarity(meaiEmbeddings[0].Vector.Span, meaiEmbeddings[2].Vector.Span);
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[1]}\": {sim01:F4}");
Console.WriteLine($"  \"{meaiTexts[0]}\" vs \"{meaiTexts[2]}\": {sim02:F4}");

// Cleanup
transformer.Dispose();
Console.WriteLine("\nDone!");

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
