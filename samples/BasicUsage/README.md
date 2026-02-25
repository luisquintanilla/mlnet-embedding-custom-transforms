# BasicUsage Sample

End-to-end demo of the **all-MiniLM-L6-v2** embedding model using all API surfaces: the composable modular pipeline, the convenience facade, chained `.Append()` pipelines, save/load round-trips, and the MEAI `IEmbeddingGenerator` interface.

## Model Details

| Property | Value |
|----------|-------|
| Model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Dimensions | 384 |
| Size | ~86 MB (ONNX) |
| Architecture | BERT-based, 6 layers |
| Use case | General-purpose semantic similarity |

## What This Sample Shows

1. **ML.NET Pipeline (facade)** — `OnnxTextEmbeddingEstimator` encapsulates tokenization → ONNX inference → pooling → normalization in a single estimator
2. **Cosine Similarity** — Pairwise similarity between embedded texts using `TensorPrimitives.CosineSimilarity`
3. **Save/Load Round-Trip** — Serialize to `.mlnet` zip, reload, and verify identical embeddings
4. **MEAI `IEmbeddingGenerator`** — Use the model through Microsoft.Extensions.AI's provider-agnostic interface
5. **Composable Pipeline (step-by-step)** — Explicit `TokenizeText → ScoreOnnxTextModel → PoolEmbedding` with individual transform inspection
6. **Chained Estimator Pipeline (`.Append`)** — Idiomatic ML.NET pattern chaining estimators, then `Fit + Transform` the whole pipeline at once

## Download Model Files

### PowerShell

```powershell
cd samples/BasicUsage
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/BasicUsage
mkdir -p models
curl -L -o models/model.onnx "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
curl -L -o models/vocab.txt "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
```

## Run

```bash
dotnet run
```

## Key Code Patterns

### Composable Pipeline (modular)

```csharp
// Each step is a separate, inspectable transform
var tokenizer = mlContext.Transforms.TokenizeText(tokenizerOpts).Fit(dataView);
var scorer = mlContext.Transforms.ScoreOnnxTextModel(scorerOpts).Fit(tokenized);
var pooler = mlContext.Transforms.PoolEmbedding(poolingOpts).Fit(scored);

// Inspect intermediate state
Console.WriteLine($"Hidden dim: {scorer.HiddenDim}");
```

### Chained Pipeline (idiomatic ML.NET)

```csharp
var pipeline = mlContext.Transforms.TokenizeText(tokenizerOpts)
    .Append(mlContext.Transforms.ScoreOnnxTextModel(scorerOpts))
    .Append(mlContext.Transforms.PoolEmbedding(poolingOpts));
var model = pipeline.Fit(dataView);
```

### Convenience Facade (single-shot)

```csharp
var estimator = new OnnxTextEmbeddingEstimator(mlContext, options);
var transformer = estimator.Fit(dataView);
```
