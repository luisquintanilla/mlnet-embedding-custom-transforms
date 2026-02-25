# ComposablePoolingComparison Sample

Same model, **three pooling strategies** — demonstrates how the modular pipeline lets you swap post-processing without re-running ONNX inference.

## Model

Uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (same model as BasicUsage).

## What This Sample Shows

This is the flagship sample for the composable pipeline architecture. It demonstrates a pattern that is **impossible with the monolithic API**: running tokenization and ONNX inference once, then applying three different pooling strategies to the same scored output.

### Pipeline Architecture

```
                                    ┌─ MeanPooling → embeddings_mean
Raw text → Tokenizer → ONNX Scorer ├─ ClsToken    → embeddings_cls
           (shared)    (shared)     └─ MaxPooling  → embeddings_max
```

### Pooling Strategies Compared

| Strategy | How it Works | Best For |
|----------|-------------|----------|
| **MeanPooling** | Average of all non-padding token embeddings | General-purpose similarity (most common) |
| **ClsToken** | Uses only the `[CLS]` token's representation | Classification-oriented models |
| **MaxPooling** | Element-wise max across all token positions | Capturing dominant features |

### Why This Needs Modularization

With the monolithic `OnnxTextEmbeddingEstimator`, changing the pooling strategy requires re-fitting the entire estimator and re-running ONNX inference. With the modular pipeline, you swap only the final pooler — the expensive ONNX inference happens once.

## Download Model Files

### PowerShell

```powershell
cd samples/ComposablePoolingComparison
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/ComposablePoolingComparison
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
// Tokenize + score ONCE (expensive ONNX inference)
var tokenized = tokenizer.Transform(dataView);
var scored = scorer.Transform(tokenized);

// Apply different pooling strategies to the SAME scored output (cheap)
foreach (var strategy in new[] { MeanPooling, ClsToken, MaxPooling })
{
    var pooler = mlContext.Transforms.PoolEmbedding(new EmbeddingPoolingOptions
    {
        Pooling = strategy,
        HiddenDim = scorer.HiddenDim,
        // ...
    }).Fit(scored);
    var embeddings = pooler.Transform(scored);
}
```
