# GTE-Small Embedding Sample

Demonstrates [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) for **semantic search without prefixes** — a straightforward embedding model that works well out of the box.

## Model Details

| Property | Value |
|----------|-------|
| Model | thenlper/gte-small |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Semantic search, similarity, clustering |

## What This Sample Shows

1. **Standard embedding** — Generate embeddings and compute pairwise cosine similarity
2. **Semantic search demo** — Rank a corpus of documents against multiple queries, showing how the model correctly identifies the most relevant passages without any prefix engineering
3. **Save/Load round-trip** — Serialize to `.mlnet` zip and verify loaded model produces identical embeddings
4. **MEAI IEmbeddingGenerator** — Use the model through Microsoft.Extensions.AI's `IEmbeddingGenerator<string, Embedding<float>>` interface

## Download Model Files

### PowerShell

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/thenlper/gte-small/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
curl -L -o models/model.onnx "https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/thenlper/gte-small/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## Why GTE-Small?

GTE (General Text Embeddings) models are designed to work well across a variety of tasks without requiring task-specific prefixes. This makes GTE-small a good default choice when you want strong embeddings with minimal configuration:

```csharp
// No prefix needed — just embed your text directly
var data = new TextData { Text = "Machine learning is a subset of AI." };
var query = new TextData { Text = "What is deep learning?" };
```

The sample's semantic search demo shows GTE-small correctly ranking relevant documents highest across different query topics.
