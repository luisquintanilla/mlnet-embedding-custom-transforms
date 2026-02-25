using System.Numerics.Tensors;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

var modelPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "model.onnx"));
var vocabPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "models", "vocab.txt"));

Console.WriteLine("=== GTE-Small Embedding Sample ===\n");

var mlContext = new MLContext();

// --- 1. Standard embedding ---
Console.WriteLine("1. Standard Embedding");
Console.WriteLine(new string('-', 40));
Console.WriteLine("  GTE-Small works well without any prefix.");

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

// --- 2. Semantic search demo ---
Console.WriteLine("\n2. Semantic Search");
Console.WriteLine(new string('-', 40));

var corpus = new[]
{
    new TextData { Text = "Machine learning is a subset of artificial intelligence." },
    new TextData { Text = "Bread baking requires flour, water, yeast, and salt." },
    new TextData { Text = "Neural networks are inspired by biological neurons." },
    new TextData { Text = "The stock market fluctuates based on economic indicators." },
    new TextData { Text = "Python and C# are popular programming languages." }
};

var corpusView = mlContext.Data.LoadFromEnumerable(corpus);
var corpusTransformed = transformer.Transform(corpusView);
var corpusEmbeddings = mlContext.Data.CreateEnumerable<EmbeddingResult>(corpusTransformed, reuseRowObject: false).ToList();

var queries = new[] { "What is deep learning?", "How do I write code?", "Tell me about bread" };
foreach (var query in queries)
{
    var queryData = new[] { new TextData { Text = query } };
    var queryView = mlContext.Data.LoadFromEnumerable(queryData);
    var queryEmbedding = mlContext.Data.CreateEnumerable<EmbeddingResult>(
        transformer.Transform(queryView), reuseRowObject: false).First();

    Console.WriteLine($"  Query: \"{query}\"");
    var ranked = corpusEmbeddings
        .Select((e, i) => (Text: corpus[i].Text, Sim: TensorPrimitives.CosineSimilarity(queryEmbedding.Embedding, e.Embedding)))
        .OrderByDescending(x => x.Sim)
        .ToList();
    foreach (var (text, sim) in ranked)
        Console.WriteLine($"    {sim:F4}  {text}");
    Console.WriteLine();
}

// --- 3. Save/Load Round-Trip ---
Console.WriteLine("3. Save/Load Round-Trip");
Console.WriteLine(new string('-', 40));

var savePath = Path.Combine(Path.GetTempPath(), "gte-small-embedding.mlnet");
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
