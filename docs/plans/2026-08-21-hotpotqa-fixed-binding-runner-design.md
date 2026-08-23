# HotpotQA Fixed-Binding Runner Design

## Context

The standard demo runner made three LLM-based Skill-selection calls for every
question. Two complete attempts failed on transient empty Manage responses,
before the already fixed `agentic-sim-rag` workflow could run. Those selection
calls are not part of the experimental variable and were not used by the
original baseline run.

## Decision

Use a dedicated HotpotQA runner that compiles one fixed plan once:

- `agentic-sim-rag`
- `component-bm25-retriever`
- `component-grounded-generator`
- `component-critic`
- no rewriter or reranker

The runner uses the tracked 20-example distractor subset and the existing
`top_k=3`, `max_iterations=3`, Generator 256-token, and Critic 4096-token
settings. It catches failures per example, records them, and continues without
retrying. Results are checkpointed after every example so a process failure does
not discard completed work.

## Alternatives rejected

- Repeatedly restarting the standard demo wastes completed calls and leaves the
  same unrelated selection failure mode.
- Increasing selection token limits changes framework-wide selection behavior
  without evidence that token exhaustion caused these two failures.

## Verification

Unit tests use a fake model and fixed demo fixtures; they assert fixed bindings,
per-example failure continuation, checkpoint output, and schema-v2 metrics.
Real API execution remains a separate command and never downloads BGE weights.
