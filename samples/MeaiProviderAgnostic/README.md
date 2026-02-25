# MeaiProviderAgnostic Sample

Use `EmbeddingGeneratorEstimator` to wrap **any** `IEmbeddingGenerator<string, Embedding<float>>` as an ML.NET transform — demonstrating **provider-agnostic** embedding generation.

## Model

Uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `OnnxEmbeddingGenerator`, but the pattern works with any [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) provider.

## What This Sample Shows

1. **Provider-Agnostic ML.NET Pipeline** — `EmbeddingGeneratorEstimator` wraps any `IEmbeddingGenerator` as a standard ML.NET transform that can participate in pipelines
2. **Cosine Similarity** — Pairwise similarity between embedded texts
3. **Direct MEAI Usage** — Use the same generator directly via `GenerateAsync` for non-pipeline scenarios

### The Provider-Agnostic Pattern

The key insight is that **only the generator construction changes** per provider. The ML.NET pipeline code stays identical:

```csharp
// --- ONNX (local) ---
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(mlContext, onnxTransformer);

// --- OpenAI ---
// IEmbeddingGenerator<string, Embedding<float>> generator =
//     new OpenAIClient(apiKey).GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator();

// --- Azure OpenAI ---
// IEmbeddingGenerator<string, Embedding<float>> generator =
//     new AzureOpenAIClient(endpoint, credential).GetEmbeddingClient("text-embedding-3-small").AsEmbeddingGenerator();

// --- Ollama ---
// IEmbeddingGenerator<string, Embedding<float>> generator =
//     new OllamaEmbeddingGenerator(new Uri("http://localhost:11434"), "all-minilm");

// Pipeline code is IDENTICAL regardless of provider
var estimator = mlContext.Transforms.TextEmbedding(generator);
var transformer = estimator.Fit(dataView);
var embeddings = transformer.Transform(dataView);
```

### Why This Needs Modularization

`EmbeddingGeneratorEstimator` is a new type introduced in proposal 05 as part of the modular architecture. It bridges the Microsoft.Extensions.AI ecosystem with ML.NET's pipeline model, enabling provider-agnostic embedding pipelines.

## Download Model Files

### PowerShell

```powershell
cd samples/MeaiProviderAgnostic
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/MeaiProviderAgnostic
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## Key Code Pattern

```csharp
// Create any IEmbeddingGenerator (provider-specific)
IEmbeddingGenerator<string, Embedding<float>> generator = ...;

// Wrap as ML.NET transform (provider-agnostic)
var estimator = mlContext.Transforms.TextEmbedding(generator);
var transformer = estimator.Fit(dataView);
var embeddings = transformer.Transform(dataView);

// Or use directly via MEAI
var results = await generator.GenerateAsync(texts);
```
