# IntermediateInspection Sample

Inspect **intermediate pipeline outputs** — token IDs, attention masks, raw ONNX model output, and final embeddings — to understand what each transform does at every stage.

## Model

Uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (same model as BasicUsage).

## What This Sample Shows

This sample walks through the modular pipeline one step at a time, printing the intermediate data at each stage. It demonstrates **inspectability** — a key motivator for the modular architecture refactor.

### Stage 1: Tokenization

After `TokenizeText`, you can inspect:
- **Token IDs** — Integer indices into the vocabulary (e.g., `[CLS]=101`, `[SEP]=102`)
- **Attention Mask** — Which positions are real tokens (1) vs. padding (0)
- **Real token count** vs. **padding count** — Shows how much of the sequence is used

### Stage 2: ONNX Scoring

After `ScoreOnnxTextEmbedding`, you can inspect:
- **Hidden dimension** — The model's internal representation size (384 for MiniLM)
- **Pre-pooled output** — Whether the model provides a `sentence_embedding` output
- **Raw output column type** — The shape and data type of the ONNX model output

### Stage 3: Pooling + Normalization

After `PoolEmbedding`, you can inspect:
- **Final embedding** — The dense float vector
- **L2 norm** — Should be ~1.0 if normalization is enabled

### Why This Needs Modularization

The monolithic `OnnxTextEmbeddingEstimator` hides all intermediate state. You can't inspect what tokens were produced, what the raw ONNX output looks like, or verify the pooling behavior. The modular pipeline exposes every stage.

## Download Model Files

### PowerShell

```powershell
cd samples/IntermediateInspection
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile "models/model.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile "models/vocab.txt"
```

### bash / curl

```bash
cd samples/IntermediateInspection
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
// Step 1: Inspect tokenization output
using var cursor = tokenized.GetRowCursor(tokenized.Schema);
var tokenIdsGetter = cursor.GetGetter<VBuffer<long>>(tokenized.Schema["TokenIds"]);
// Print token IDs, attention masks, padding counts...

// Step 2: Inspect scorer metadata
Console.WriteLine($"Hidden dim: {scorer.HiddenDim}");
Console.WriteLine($"Pre-pooled: {scorer.HasPooledOutput}");

// Step 3: Verify normalization
var norm = MathF.Sqrt(embedding.Sum(x => x * x));
Console.WriteLine($"L2 norm: {norm:F4}"); // should be ~1.0
```
