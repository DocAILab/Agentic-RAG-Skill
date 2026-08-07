---
name: agentic-vanilla-rag
description: Arrange a sequential RAG workflow with one required retriever, one optional reranker, and one required grounded generator. Use after Manage selects a simple one-route retrieval and generation process.
---

# Vanilla RAG Workflow

Use this Skill only after the Manage stage selects it. This Skill defines process order and Component selection guidance; it does not implement retrieval, reranking, or generation.

## Component selection

1. Select exactly one `retriever` Component.
2. Optionally select one `reranker` Component when first-stage ranking is noisy.
3. Select exactly one `generator` Component.
4. Load only the selected Component Skill bodies and implementations.

Prefer BM25 for exact names and lexical anchors. Prefer vector retrieval for paraphrases and semantic matching.

## Workflow

Execute `scripts/workflow.py:run` after every required slot is bound:

```text
RAGRequest
  -> retriever
  -> optional reranker
  -> generator
  -> RAGResult
```

The workflow script may call slots through `components`, but it must never import a concrete Component implementation.
