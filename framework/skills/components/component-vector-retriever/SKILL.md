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

The default is `top_k=10`. The runtime embedding configuration used for the
validated BGE run uses `BAAI/bge-large-en-v1.5`, normalized vectors, and
`batch_size=8`; callers may supply another compatible embedding service.

Output `RetrievalResult`:

- `documents`: ranked document dictionaries with cosine `score`

## Execution

Run `scripts/component.py:run(inputs, context)`. The runtime must provide `context.embed(texts)`. When the runtime additionally provides `context.search_vector_index(...)`, the Component uses its persistent corpus index; otherwise it falls back to direct embedding and cosine scoring. This optional acceleration keeps the Skill executable in Claude Code and other Agent environments that only implement the basic embedding contract. This Component performs similarity retrieval only and does not select a model or another Skill.

The optional `run.prepare(inputs, context)` hook prepares the selected Task's
Corpus through `context.prepare_corpus_vectors(...)` before any QA starts. Keep
this hook and query retrieval on the same document formatter and format version.
Preparation encodes no query. The runtime shares the resulting index across all
QA workers and later Tasks with the same Corpus, embedding model and document
representation. With a configured disk cache, later runs load a validated index.
Contexts exposing only the basic embedding contract can still execute `run`.
