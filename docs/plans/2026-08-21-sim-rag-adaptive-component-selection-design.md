# SIM-RAG Adaptive Component Selection Design

## Goal

Validate that `agentic-sim-rag` can guide per-question Component selection while
keeping the Agentic Skill fixed. The change must not modify any Component
implementation and must preserve the existing fixed-BM25 experiment as a
control baseline.

## Selected approach

Reuse the framework's existing `select_component_skills()` stage. Load
`agentic-sim-rag` directly as the selected Agentic Skill, give its complete
instructions and compatible Component advertisements to the Executor Model,
validate the returned bindings, compile the selected plan, and execute it.

This keeps selection in the framework's Agentic-to-Component boundary rather
than adding rule-based selection to the experiment or recompiling Components
inside every workflow iteration.

## Selection policy

`agentic-sim-rag/SKILL.md` will provide concise guidance:

- Prefer BM25 for exact names, dates, titles, identifiers, and strong lexical
  overlap.
- Prefer Vector retrieval for paraphrases and semantic matching.
- Use HyDE only with Vector retrieval when a hypothetical passage is likely to
  bridge vocabulary or intent gaps.
- Use BGE Reranker when multi-hop or noisy candidate evidence makes top-rank
  precision important.
- Always bind the required Generator and Critic slots.
- Use optional Components only when their expected benefit justifies their
  latency and model cost.

The framework remains responsible for slot cardinality and dependency checks.
In particular, the existing HyDE requirement on Vector retrieval is preserved.

## Adaptive experiment flow

For every HotpotQA example:

1. Build the normal RAG request and candidate-document collection.
2. Load `agentic-sim-rag` as the fixed Agentic selection.
3. Ask the Executor Model to select compatible Components for that request.
4. Validate and compile the selected bindings.
5. Execute the iterative workflow.
6. Record bindings, selection reason, answer, retrieval trace, and metrics.

The report will contain per-example selection data and aggregate binding
frequencies. Selection, compilation, and execution failures will retain their
stage so configuration problems can be distinguished from RAG failures.

## Files and interfaces

- Update `framework/skills/agentic/agentic-sim-rag/SKILL.md` with selection
  guidance.
- Add a separate adaptive HotpotQA runner and example configuration under
  `experiments/hotpotqa/`.
- Keep the current fixed-binding runner and its report schema intact.
- Reuse public framework selection and compilation interfaces; do not import
  concrete Component implementations.

## Runtime prerequisites

The adaptive configuration will enable an Embedding client so a selected Vector
Retriever can execute. BGE Reranker and local Embedding execution require the
existing optional `sentence-transformers` dependency and model weights. Unit
tests will use fakes and will not download weights or call external APIs.

## Test strategy

Use vertical RED-GREEN-REFACTOR cycles to verify:

1. The fixed `agentic-sim-rag` instructions can drive dynamic Component
   selection and execution through public framework APIs.
2. Different examples may select different valid bindings.
3. Per-example bindings and reasons are written to the adaptive report.
4. Invalid dependency selections fail at the selection stage and are recorded.
5. The existing fixed-BM25 runner remains unchanged and passing.

After implementation, run targeted tests, the full test suite, and Ruff. Do not
run the real HotpotQA adaptive experiment until the user reviews the code and
explicitly approves the run.

## Out of scope

- Modifying Component code or contracts.
- Supporting multiple Retriever bindings in one `agentic-sim-rag` plan.
- Adding regenerate-vs-retrieve Critic routing in the same change.
- Claiming adaptive-selection quality before a real controlled experiment.
