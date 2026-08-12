---
name: component-grounded-generator
description: Generate an answer grounded in selected documents by calling the frozen Executor Model through the runtime context. Use as the final generator in Agentic RAG workflows.
---

# Grounded Generator Component

Provide the `generator` capability for an Agentic RAG slot.

## Interface

Input `GenerationRequest`:

- `query`: original question
- `documents`: selected evidence documents
- optional `max_tokens`

Output `GenerationResult`:

- `answer`: generated grounded answer

## Execution

Run `scripts/component.py:run(inputs, context)`. The runtime must provide the frozen `context.call_model(...)`. Do not retrieve documents or modify model parameters.
