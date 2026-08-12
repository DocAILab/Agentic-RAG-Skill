---
name: agentic-rrfusion
description: Arrange parallel retrieval with two to four distinct retrievers, Reciprocal Rank Fusion, optional reranking, and grounded generation. Use when multiple retrieval signals should improve recall or robustness.
---

# RRFusion Workflow

Use this Skill only after the Manage stage selects it. Keep retrieval implementations outside the workflow script.

## Component selection

1. Select two to four distinct `retriever` Components.
2. Prefer Component diversity, such as BM25 plus Vector Retrieval.
3. Optionally select one `reranker` after fusion.
4. Select exactly one `generator` Component.
5. Load only selected Component bodies and implementations.

## Workflow

Execute `scripts/workflow.py:run` after binding every required slot:

```text
RAGRequest
  -> retrievers in parallel
  -> reciprocal rank fusion
  -> optional reranker
  -> generator
  -> RAGResult
```

The workflow owns fusion and ordering. Each retriever owns only its concrete retrieval algorithm.
