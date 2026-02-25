# E5-Small Embedding Sample

Demonstrates [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) with the **query/passage prefix pattern** used in E5 embedding models.

## Model Details

| Property | Value |
|----------|-------|
| Model | intfloat/e5-small-v2 |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Retrieval, semantic similarity |

## What This Sample Shows

1. **Standard embedding** — Generate embeddings and compute cosine similarity between texts
2. **Query/passage prefixes for retrieval** — E5 models use `"query: "` for queries and `"passage: "` for documents. The sample compares similarity scores with and without prefixes to show their effect on retrieval ranking.
3. **Save/Load round-trip** — Serialize to `.mlnet` zip and verify loaded model produces identical embeddings
4. **MEAI IEmbeddingGenerator** — Use the model through Microsoft.Extensions.AI's `IEmbeddingGenerator<string, Embedding<float>>` interface

## Download Model Files

### PowerShell

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/intfloat/e5-small-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
curl -L -o models/model.onnx "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/intfloat/e5-small-v2/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## E5 Prefix Pattern

E5 models use distinct prefixes for queries and passages. Both sides must be prefixed for optimal retrieval:

```csharp
// Passages — use "passage: " prefix
var passage = new TextData { Text = "passage: Machine learning is a subset of AI." };

// Queries — use "query: " prefix
var query = new TextData { Text = "query: What is AI?" };
```

This dual-prefix scheme helps the model learn separate representations for questions vs. documents, improving asymmetric retrieval. The sample demonstrates how using the correct prefixes affects similarity scores compared to unprefixed text.
