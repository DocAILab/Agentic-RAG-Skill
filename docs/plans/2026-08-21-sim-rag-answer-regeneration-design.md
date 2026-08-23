# SIM-RAG Answer Regeneration Design

## Goal

Improve `agentic-sim-rag` when the Critic rejects an answer even though the
available evidence is sufficient. Distinguish answer-form problems from
missing-evidence problems, regenerate a concise answer for the former, and
continue retrieval for the latter.

The change must remain inside the Agentic workflow. The existing Critic and
Generator Component interfaces remain unchanged.

## Considered approaches

1. Extend `CritiqueResult` with an explicit `approve/retrieve/regenerate`
   decision. This is the most reliable option, but changes a teammate-owned
   Component contract.
2. Add a new answer-revision Component and Agentic slot. This preserves the
   existing Components but expands selection, compilation, and binding scope.
3. Classify the existing Critic feedback conservatively inside the Agentic
   workflow. This has weaker semantic guarantees than an explicit enum, but it
   is compatible with every existing Component and is the approved approach.

## Decision policy

The workflow inspects normalized Critic `feedback` and `issues` after an
unapproved candidate:

- Evidence-gap language such as missing, unsupported, not established, or a
  missing reasoning hop means `retrieve`.
- Answer-form language such as too verbose, not direct, formatting, wording,
  or extra explanation means `regenerate` only when no evidence-gap signal is
  present.
- Mixed or unrecognized feedback defaults to `retrieve`.

This conservative fallback prevents an ambiguous rejection from bypassing
evidence gathering.

## Workflow

Each iteration keeps the existing retrieve, merge, rerank, generate, and
critique steps. If the first critique is approved, the workflow returns as
before. If it is classified as `retrieve`, the workflow continues with the
existing feedback query.

If it is classified as `regenerate`, the workflow calls the bound Generator
once more with the same accumulated evidence. The revision query contains the
original question plus bounded Critic guidance requesting the shortest direct
answer. The revised answer is sent through the Critic again:

- approved: return it with stop reason
  `critic_approved_after_regeneration`;
- rejected: use the revised critique for stopping or follow-up retrieval.

At most one regeneration is allowed per iteration. Existing maximum-iteration,
no-new-evidence, and safe-answer behavior remains intact.

## Trace

When regeneration occurs, append a `regeneration` event after the iteration
event. It records the iteration number, original candidate, bounded revision
guidance, revised answer, and complete revised Critic result. This separates
answer repair from evidence acquisition in experiment logs.

## Tests

- A format-only rejection regenerates and can stop without another retrieval.
- A missing-evidence rejection continues retrieval without regeneration.
- Mixed or ambiguous feedback conservatively continues retrieval.
- A rejected revised answer is not returned and follows existing safe stopping.
- Existing approval, HyDE, reranking, token-budget, and trace tests remain green.

