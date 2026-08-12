---
name: component-vector-retriever
description: Retrieve and rank documents by cosine similarity using the embedding service supplied by the runtime context. Use for paraphrases, semantic matching, and weak lexical overlap.
---

# Vector Retriever Component

Provide the `retriever` capability for an Agentic RAG slot.

## Interface

Input `RetrievalRequest`:

- `query`: retrieval query
- `documents`: JSON-compatible documents containing `id` and `text`
- `top_k`: maximum returned documents

Output `RetrievalResult`:

- `documents`: ranked document dictionaries with cosine `score`

## Execution

Run `scripts/component.py:run(inputs, context)`. The runtime must provide `context.embed(texts)`. This Component performs similarity retrieval only and does not select a model or another Skill.
