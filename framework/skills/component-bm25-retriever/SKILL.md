---
name: component-bm25-retriever
description: Retrieve and rank documents with a concrete Okapi BM25 implementation. Use for exact names, identifiers, quotations, rare terms, and queries with reliable lexical overlap.
---

# BM25 Retriever Component

Provide the `retriever` capability for an Agentic RAG slot.

## Interface

Input `RetrievalRequest`:

- `query`: retrieval query
- `documents`: JSON-compatible documents containing `id` and `text`
- `top_k`: maximum returned documents
- optional `k1` and `b`

Output `RetrievalResult`:

- `documents`: ranked document dictionaries with BM25 `score`

## Execution

Run `scripts/component.py:run(inputs, context)`. This Component performs retrieval only. Do not choose other Skills or generate an answer.
