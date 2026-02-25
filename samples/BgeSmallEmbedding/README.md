# BGE-Small Embedding Sample

Demonstrates [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) with the **composable modular pipeline** and the **query prefix pattern** used in BGE retrieval models.

## Model Details

| Property | Value |
|----------|-------|
| Model | BAAI/bge-small-en-v1.5 |
| Dimensions | 384 |
| Size | ~127 MB (ONNX) |
| Architecture | BERT-based |
| Use case | Retrieval, semantic similarity |

## What This Sample Shows

1. **Composable Modular Pipeline** — Explicit `TokenizeText → ScoreOnnxTextModel → PoolEmbedding` steps using directory-based tokenizer auto-detection
2. **Query prefix for retrieval** — BGE models recommend prepending `"Represent this sentence: "` to queries when doing retrieval against a passage corpus. The sample compares similarity scores with and without the prefix to show its effect
3. **Chained Estimator Pipeline (`.Append`)** — Idiomatic ML.NET pattern with all three transforms chained
4. **Convenience Facade** — `OnnxTextEmbeddingEstimator` as a single-shot alternative, verified to produce identical results

## Download Model Files

### PowerShell

```powershell
cd samples/BgeSmallEmbedding
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/BgeSmallEmbedding
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

## Composable Pipeline Pattern

This sample demonstrates the modular pipeline where each step is a separate, inspectable transform:

```csharp
// Each transform can be inspected and reused independently
var tokenizer = mlContext.Transforms.TokenizeText(tokenizerOpts).Fit(dataView);
var scorer = mlContext.Transforms.ScoreOnnxTextModel(scorerOpts).Fit(tokenized);
var pooler = mlContext.Transforms.PoolEmbedding(poolingOpts).Fit(scored);

// Reuse the fitted pipeline for multiple embedding calls
var passageEmbeddings = Embed(passages);
var queryEmbeddings = Embed(queries);
```
