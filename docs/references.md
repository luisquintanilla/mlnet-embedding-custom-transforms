# References & Further Reading

A curated list of all sources used in the design and implementation of this project, organized by topic.

## ML.NET Custom Transform Guidance

- **ML.NET Custom Transformer Guide** — The primary design reference for understanding the four approaches (A–D) to building custom transforms.
  - DeepWiki: https://deepwiki.com/luisquintanilla/mlnet-custom-transformer-guide
  - Source repo: https://github.com/luisquintanilla/mlnet-custom-transformer-guide

- **ML.NET Source Code — OnnxTransformer** — Reference implementation for how ML.NET wraps OnnxRuntime internally. Used as a pattern for auto-discovery of input/output tensor names.
  - https://github.com/dotnet/machinelearning/tree/main/src/Microsoft.ML.OnnxTransformer

- **ML.NET Source Code — FeaturizeTextTransform** — Reference for how ML.NET encapsulates multiple transforms into a single convenience API. Inspired our single-shot design.
  - https://github.com/dotnet/machinelearning/blob/main/src/Microsoft.ML.Transforms/Text/TextFeaturizingEstimator.cs

- **ML.NET Official Documentation** — General ML.NET concepts, IDataView architecture, and transform patterns.
  - https://learn.microsoft.com/en-us/dotnet/machine-learning/

## System.Numerics.Tensors

- **dotnet-tensors-guide** — Comprehensive guide to `Tensor<T>` and `TensorPrimitives` in .NET.
  - DeepWiki: https://deepwiki.com/luisquintanilla/dotnet-tensors-guide
  - Source repo: https://github.com/luisquintanilla/dotnet-tensors-guide

- **System.Numerics.Tensors API Reference** — Official API documentation for `Tensor<T>`, `TensorPrimitives`, and `TensorSpan<T>`.
  - https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors

- **TensorPrimitives API Reference** — SIMD-accelerated math operations on spans.
  - https://learn.microsoft.com/en-us/dotnet/api/system.numerics.tensors.tensorprimitives

## ONNX Runtime

- **OnnxRuntime C# API Reference** — `InferenceSession`, `OrtValue`, `SessionOptions`, and related types.
  - https://onnxruntime.ai/docs/api/csharp-api.html

- **OnnxRuntime NuGet Package** — `Microsoft.ML.OnnxRuntime` (v1.24.2 used in this project).
  - https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime

## Tokenization

- **Microsoft.ML.Tokenizers** — .NET tokenizer library supporting BPE, WordPiece (BERT), SentencePiece, and Tiktoken.
  - https://www.nuget.org/packages/Microsoft.ML.Tokenizers
  - API Reference: https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers

- **BertTokenizer** — The specific tokenizer class used for all-MiniLM-L6-v2 (WordPiece/BERT vocab.txt format).
  - https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers.berttokenizer

## Embedding Models

- **all-MiniLM-L6-v2** — The primary model used in this project. 384-dimensional sentence embeddings, 80MB ONNX. Excellent quality/size ratio.
  - HuggingFace model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  - ONNX variant: https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main/onnx

- **Sentence-Transformers** — The framework that produced most popular embedding models. Explains pooling strategies, training objectives, and evaluation.
  - https://www.sbert.net/

- **MTEB Leaderboard** — Massive Text Embedding Benchmark. Compare embedding models across tasks (retrieval, classification, clustering, etc.).
  - https://huggingface.co/spaces/mteb/leaderboard

### Other Compatible Models

| Model | Dimensions | Size | HuggingFace |
|-------|-----------|------|-------------|
| all-MiniLM-L6-v2 | 384 | ~80MB | [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| all-mpnet-base-v2 | 768 | ~420MB | [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| e5-small-v2 | 384 | ~80MB | [link](https://huggingface.co/intfloat/e5-small-v2) |
| bge-small-en-v1.5 | 384 | ~80MB | [link](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| paraphrase-MiniLM-L12-v2 | 384 | ~120MB | [link](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2) |

## Microsoft.Extensions.AI (MEAI)

- **MEAI Abstractions NuGet Package** — `Microsoft.Extensions.AI.Abstractions` (v10.3.0 used in this project). Defines `IEmbeddingGenerator<TInput, TEmbedding>`.
  - https://www.nuget.org/packages/Microsoft.Extensions.AI.Abstractions

- **MEAI Documentation** — Official documentation for the unified AI abstractions in .NET.
  - https://learn.microsoft.com/en-us/dotnet/ai/ai-extensions

- **IEmbeddingGenerator Interface** — The abstraction we implement via `OnnxEmbeddingGenerator`.
  - https://learn.microsoft.com/en-us/dotnet/api/microsoft.extensions.ai.iembeddinggenerator-2

## Existing Implementations (Inspiration)

- **MLNETEmbeddingGenerator** — An older implementation of ML.NET with embedding generation. Used as a reference for understanding the problem space.
  - https://github.com/luisquintanilla/MLNETEmbeddingGenerator

## .NET Platform

- **.NET 10** — Target framework for this project.
  - https://learn.microsoft.com/en-us/dotnet/core/whats-new/dotnet-10/overview

- **System.IO.Compression** — Used in `ModelPackager` for creating self-contained zip archives.
  - https://learn.microsoft.com/en-us/dotnet/api/system.io.compression

- **System.Text.Json Source Generation** — Used in `ModelPackager` for AOT-compatible JSON serialization of config and manifest.
  - https://learn.microsoft.com/en-us/dotnet/standard/serialization/system-text-json/source-generation
