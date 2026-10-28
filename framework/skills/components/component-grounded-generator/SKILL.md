---
name: component-grounded-generator
description: Generate an answer grounded in selected documents by calling the frozen Executor Model through the runtime context. Use as the final generator in Agentic RAG workflows.
---

# Grounded Generator Component

Provide the `generator` capability for an Agentic RAG slot.

## Interface

Input `GenerationRequest`:

- `query`: non-empty question, optionally including answer revision guidance
- `documents`: selected evidence documents. Each document must contain non-empty
  `id` and `text`; an empty sequence is allowed and is rendered as
  `[No evidence supplied]`.
- optional `max_tokens`: forwarded to the frozen Executor Model
- optional `evidence_query`: non-empty original question used to select evidence
  windows; defaults to `query`. Preserve it when adding answer revision guidance.

Output `GenerationResult`:

- `answer`: non-empty generated grounded answer. Leading and trailing whitespace
  from the model response is stripped.

## Execution

Run `scripts/component.py:run(inputs, context)`. The runtime must provide the
frozen `context.call_model(...)`. The component formats the supplied evidence,
calls the model with `temperature=0.0`, and rejects malformed inputs or empty
model responses. Shared `framework.prepare_evidence` preserves short documents
and selects query-matching windows from long documents, within 4 KiB per document
and 96 KiB total UTF-8 bytes including evidence formatting. Omitted text and
documents are explicitly marked. When supported by the runtime, document IDs,
original character offsets and byte counts are recorded in component timings.
Do not retrieve documents, rerank documents, or modify model parameters.
