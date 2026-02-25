# E5-Small Embedding Sample

Demonstrates [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) with the **composable modular pipeline** and the **dual query/passage prefix pattern** used in E5 embedding models.

## Model Details

| Property | Value |
|----------|-------|
| Model | intfloat/e5-small-v2 |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Retrieval, semantic similarity |

## What This Sample Shows

1. **Composable Modular Pipeline** — Explicit `TokenizeText → ScoreOnnxTextModel → PoolEmbedding` steps using directory-based tokenizer auto-detection
2. **Query/passage prefixes for retrieval** — E5 models use `"query: "` for queries and `"passage: "` for documents. The sample compares similarity scores with and without prefixes to show their effect on retrieval ranking
3. **Chained Estimator Pipeline (`.Append`)** — Idiomatic ML.NET pattern with all three transforms chained
4. **Convenience Facade** — `OnnxTextEmbeddingEstimator` as a single-shot alternative, verified to produce identical results

## Download Model Files

### PowerShell

```powershell
cd samples/E5SmallEmbedding
Invoke-WebRequest -Uri "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/intfloat/e5-small-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/E5SmallEmbedding
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

## Composable Pipeline Pattern

This sample demonstrates the modular pipeline where each step is a separate, inspectable transform:

```csharp
// Build the pipeline step by step
var tokenizer = mlContext.Transforms.TokenizeText(tokenizerOpts).Fit(dataView);
var scorer = mlContext.Transforms.ScoreOnnxTextModel(scorerOpts).Fit(tokenized);
var pooler = mlContext.Transforms.PoolEmbedding(poolingOpts).Fit(scored);

// Inspect model metadata
Console.WriteLine($"Hidden dim: {scorer.HiddenDim}");
Console.WriteLine($"Pre-pooled: {scorer.HasPooledOutput}");
```
