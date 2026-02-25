# GTE-Small Embedding Sample

Demonstrates [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) with the **composable modular pipeline** for **semantic search without prefixes** — a straightforward embedding model that works well out of the box.

## Model Details

| Property | Value |
|----------|-------|
| Model | thenlper/gte-small |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Semantic search, similarity, clustering |

## What This Sample Shows

1. **Composable Modular Pipeline** — Explicit `TokenizeText → ScoreOnnxTextModel → PoolEmbedding` steps using directory-based tokenizer auto-detection
2. **Semantic search demo** — Rank a corpus of documents against multiple queries, showing how the model correctly identifies the most relevant passages without any prefix engineering
3. **Chained Estimator Pipeline (`.Append`)** — Idiomatic ML.NET pattern with all three transforms chained
4. **Convenience Facade** — `OnnxTextEmbeddingEstimator` as a single-shot alternative, verified to produce identical results

## Download Model Files

### PowerShell

```powershell
cd samples/GteSmallEmbedding
Invoke-WebRequest -Uri "https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/thenlper/gte-small/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/GteSmallEmbedding
curl -L -o models/model.onnx "https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/thenlper/gte-small/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## Why GTE-Small?

GTE (General Text Embeddings) models are designed to work well across a variety of tasks without requiring task-specific prefixes. This makes GTE-Small a good default choice when you want strong embeddings with minimal configuration:

```csharp
// No prefix needed — just embed your text directly
var data = new TextData { Text = "Machine learning is a subset of AI." };
var query = new TextData { Text = "What is deep learning?" };
```

The sample's semantic search demo shows GTE-Small correctly ranking relevant documents highest across different query topics.

## Composable Pipeline Pattern

This sample uses the modular pipeline with reusable fitted transforms:

```csharp
var tokenizer = mlContext.Transforms.TokenizeText(tokenizerOpts).Fit(dataView);
var scorer = mlContext.Transforms.ScoreOnnxTextModel(scorerOpts).Fit(tokenized);
var pooler = mlContext.Transforms.PoolEmbedding(poolingOpts).Fit(scored);

// Helper function reuses the fitted pipeline for multiple batches
IList<EmbeddingResult> Embed(TextData[] texts) =>
    pooler.Transform(scorer.Transform(tokenizer.Transform(dv)));
```
