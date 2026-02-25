# MLNet.Embeddings.Onnx

A custom ML.NET `IEstimator` / `ITransformer` that generates text embeddings using local HuggingFace ONNX models. One call encapsulates the entire pipeline — tokenization, ONNX inference, pooling, and normalization — just like `FeaturizeText` does for classical text features.

```
Raw text  →  [BertTokenizer]  →  token IDs + attention mask
          →  [OnnxRuntime]    →  last_hidden_state
          →  [Mean Pooling]   →  L2-normalized float[384] embedding
```

## Why This Exists

ML.NET has no built-in transform for modern HuggingFace embedding models (all-MiniLM-L6-v2, BGE, E5, etc.). Building one is hard because ML.NET's convenient internal base classes (`RowToRowTransformerBase`, `OneToOneTransformerBase`) have `private protected` constructors — they can't be subclassed from external projects.

This project implements a custom transform using direct `IEstimator<T>` / `ITransformer` interfaces (Approach C from the [ML.NET Custom Transformer Guide](https://github.com/luisquintanilla/mlnet-custom-transformer-guide)), enhanced with custom zip-based save/load for model persistence.

## Features

- **Single-shot API** — one estimator encapsulates tokenization → ONNX inference → pooling → normalization
- **ONNX auto-discovery** — automatically detects input/output tensor names, shapes, and embedding dimensions from model metadata
- **Self-contained save/load** — serializes to a portable `.mlnet` zip file containing the ONNX model, tokenizer, and config
- **MEAI integration** — includes an `IEmbeddingGenerator<string, Embedding<float>>` wrapper for use with Microsoft.Extensions.AI
- **SIMD-accelerated pooling** — mean pooling and L2 normalization use `TensorPrimitives` for hardware-vectorized math
- **Configurable batching** — process rows in configurable batch sizes to bound memory usage
- **Multiple pooling strategies** — Mean, CLS token, and Max pooling

## Quickstart

### 1. Get the model files

Download [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) ONNX model and vocabulary:

```powershell
mkdir models
# ONNX model (~86 MB)
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
# Vocabulary file
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### 2. Use as an ML.NET transform

```csharp
using Microsoft.ML;
using MLNet.Embeddings.Onnx;

var mlContext = new MLContext();

var estimator = new OnnxTextEmbeddingEstimator(mlContext, new OnnxTextEmbeddingOptions
{
    ModelPath = "models/model.onnx",
    TokenizerPath = "models/vocab.txt",
});

var data = mlContext.Data.LoadFromEnumerable(new[]
{
    new { Text = "What is machine learning?" },
    new { Text = "How to cook pasta" }
});

var transformer = estimator.Fit(data);
var embeddings = transformer.Transform(data);
```

### 3. Use as an MEAI embedding generator

```csharp
using Microsoft.Extensions.AI;
using MLNet.Embeddings.Onnx;

IEmbeddingGenerator<string, Embedding<float>> generator =
    new OnnxEmbeddingGenerator(new MLContext(), transformer);

var results = await generator.GenerateAsync(["What is .NET?", "Tell me about C#"]);
float similarity = TensorPrimitives.CosineSimilarity(
    results[0].Vector.Span, results[1].Vector.Span);
```

### 4. Save and load

```csharp
// Save — bundles ONNX model + tokenizer + config into a portable zip
transformer.Save("my-embedding-model.mlnet");

// Load — fully self-contained, no external file dependencies
var loaded = OnnxTextEmbeddingTransformer.Load(mlContext, "my-embedding-model.mlnet");
```

## Project Structure

```
mlnet-embedding-custom-transforms/
├── src/MLNet.Embeddings.Onnx/
│   ├── OnnxTextEmbeddingOptions.cs      — Configuration POCO with smart defaults
│   ├── PoolingStrategy.cs               — Mean / CLS / Max pooling enum
│   ├── OnnxTextEmbeddingEstimator.cs    — IEstimator: validates model, auto-discovers tensors
│   ├── OnnxTextEmbeddingTransformer.cs  — ITransformer: tokenize → infer → pool → embed
│   ├── EmbeddingPooling.cs              — SIMD-accelerated pooling via TensorPrimitives
│   ├── ModelPackager.cs                 — Save/load to self-contained zip
│   ├── OnnxEmbeddingGenerator.cs        — MEAI IEmbeddingGenerator wrapper
│   └── MLContextExtensions.cs           — Convenience extension method
├── samples/BasicUsage/
│   ├── Program.cs                       — End-to-end demo with all 4 usage patterns
│   └── models/                          — Downloaded ONNX model + vocab
├── docs/                                — Detailed documentation
│   ├── design-decisions.md              — Why every choice was made
│   ├── architecture.md                  — Component walkthrough + pipeline stages
│   ├── tensor-deep-dive.md              — System.Numerics.Tensors for AI workloads
│   ├── extending.md                     — How to modify and extend
│   └── references.md                    — All sources and further reading
└── nuget.config                         — NuGet source (nuget.org only)
```

## API at a Glance

| Class | Role | Key Methods |
|-------|------|-------------|
| `OnnxTextEmbeddingEstimator` | ML.NET `IEstimator<T>` | `Fit(IDataView)`, `GetOutputSchema()` |
| `OnnxTextEmbeddingTransformer` | ML.NET `ITransformer` | `Transform(IDataView)`, `Save(path)`, `Load(ctx, path)` |
| `OnnxEmbeddingGenerator` | MEAI `IEmbeddingGenerator` | `GenerateAsync(texts)` |
| `OnnxTextEmbeddingOptions` | Configuration | `ModelPath`, `TokenizerPath`, `Pooling`, `Normalize`, `BatchSize` |

## Supported Models

Any sentence-transformer ONNX model that follows the standard input/output convention:

| Model | Dimensions | Size | Tested |
|-------|:----------:|:----:|:------:|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | ~86 MB | ✅ |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 384 | ~120 MB | — |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 768 | ~420 MB | — |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | ~80 MB | — |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | 384 | ~80 MB | — |

Models with `sentence_embedding` output (pre-pooled) are auto-detected and pooling is skipped.

## NuGet Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.ML` | 5.0.0 | IEstimator/ITransformer, IDataView, MLContext |
| `Microsoft.ML.OnnxRuntime` | 1.24.2 | InferenceSession, OrtValue |
| `Microsoft.ML.Tokenizers` | 2.0.0 | BertTokenizer (WordPiece) |
| `Microsoft.Extensions.AI.Abstractions` | 10.3.0 | IEmbeddingGenerator |
| `System.Numerics.Tensors` | 10.0.3 | Tensor\<T\>, TensorPrimitives |

## Documentation

For detailed documentation on the design, architecture, and implementation:

- **[Design Decisions](docs/design-decisions.md)** — Why every choice was made
- **[Architecture](docs/architecture.md)** — Component walkthrough and pipeline stages
- **[Tensor Deep Dive](docs/tensor-deep-dive.md)** — System.Numerics.Tensors for AI workloads
- **[Extending](docs/extending.md)** — How to modify, extend, and harden
- **[References](docs/references.md)** — All sources and further reading

## Target Framework

.NET 10 (LTS). Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).

## License

This is a prototype / reference implementation for educational purposes.
