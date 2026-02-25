# BGE-Small Embedding Sample

Demonstrates [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) with the **query prefix pattern** used in BGE retrieval models.

## Model Details

| Property | Value |
|----------|-------|
| Model | BAAI/bge-small-en-v1.5 |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Retrieval, semantic similarity |

## What This Sample Shows

1. **Standard embedding** — Generate embeddings and compute cosine similarity between texts
2. **Query prefix for retrieval** — BGE models recommend prepending `"Represent this sentence: "` to queries when doing retrieval against a passage corpus. The sample compares similarity scores with and without the prefix to show its effect.
3. **Save/Load round-trip** — Serialize to `.mlnet` zip and verify loaded model produces identical embeddings
4. **MEAI IEmbeddingGenerator** — Use the model through Microsoft.Extensions.AI's `IEmbeddingGenerator<string, Embedding<float>>` interface

## Download Model Files

### PowerShell

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
curl -L -o models/model.onnx "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## BGE Query Prefix Pattern

BGE models are trained with an instruction-aware objective. For retrieval tasks, queries should be prefixed while passages should not:

```csharp
// Passages — no prefix
var passage = new TextData { Text = "Machine learning is a subset of AI." };

// Queries — add prefix for retrieval
var query = new TextData { Text = "Represent this sentence: What is AI?" };
```

This prefix helps the model distinguish between queries and documents, improving retrieval ranking. The sample demonstrates the measurable difference in cosine similarity scores when using the prefix vs. not.
