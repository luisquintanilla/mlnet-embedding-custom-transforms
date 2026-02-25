# Proposal: Modular Transform Pipeline

## Summary

Decompose `OnnxTextEmbeddingTransformer` into three composable ML.NET transforms that form a reusable foundation for **any** transformer-based ONNX model task, not just embedding generation.

| Transform | Responsibility | Reusability |
|-----------|---------------|-------------|
| `TextTokenizerTransformer` | Text → token IDs + attention mask | Any transformer model |
| `OnnxTextModelScorerTransformer` | Token columns → raw ONNX output | Any transformer ONNX model |
| `EmbeddingPoolingTransformer` | Raw output → pooled embedding | Embedding generation |

The existing `OnnxTextEmbeddingEstimator` becomes a convenience facade that chains all three, preserving the current API.

## Motivation

1. **Composability**: ML.NET's design is composable pipelines of single-responsibility transforms. The monolith violates this.
2. **Reusability**: Tokenization and model scoring are universal — every transformer task (classification, NER, QA, reranking) starts with tokenized text fed through an ONNX model. Only the post-processing differs.
3. **Inspectability**: Users can inspect intermediate results (what tokens were produced? what does the raw model output look like?).
4. **Extensibility**: Adding a new task (e.g., text classification) requires only a new post-processing transform, not a new end-to-end pipeline.
5. **Testability**: Each transform can be unit-tested in isolation.

## Architecture

See [architecture.md](architecture.md) for the full component diagram and data flow.

## Detailed Specifications

- [01-text-tokenizer-transform.md](01-text-tokenizer-transform.md) — TextTokenizerEstimator / TextTokenizerTransformer
- [02-onnx-text-model-scorer-transform.md](02-onnx-text-model-scorer-transform.md) — OnnxTextModelScorerEstimator / OnnxTextModelScorerTransformer
- [03-embedding-pooling-transform.md](03-embedding-pooling-transform.md) — EmbeddingPoolingEstimator / EmbeddingPoolingTransformer
- [04-facade-refactor.md](04-facade-refactor.md) — OnnxTextEmbeddingEstimator / OnnxTextEmbeddingTransformer refactoring
- [05-meai-integration.md](05-meai-integration.md) — OnnxEmbeddingGenerator and IEmbeddingGenerator-backed transform

## Implementation Order

See [implementation-plan.md](implementation-plan.md) for the ordered task list with dependencies and acceptance criteria.

## Tradeoffs

See [tradeoffs.md](tradeoffs.md) for a detailed analysis of what this costs and what it gains.

## Future Task Expansion

See [future-tasks.md](future-tasks.md) for how the reusable foundation enables classification, NER, QA, and other transformer tasks.
