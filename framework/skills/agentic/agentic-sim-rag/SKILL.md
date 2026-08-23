---
name: agentic-sim-rag
description: Use when a RAG request needs bounded multi-round evidence gathering with Critic-based sufficiency checks before returning an answer.
---

# SIM-RAG-Inspired Iterative RAG

Use this immutable Agentic Skill to run a bounded retrieve-generate-critique
loop. It borrows SIM-RAG's inference architecture, but does not reproduce its
Self-Practicing, Critic training, rationale generation, or experiments.

## Contract

- Bind exactly one retriever, generator, and critic.
- Optionally bind one rewriter and one reranker.
- Keep the user's original query for reranking, generation, and critique.
- Treat rewritten queries as retrieval aids, never as evidence.
- Return an answer only when the Critic approves a direct, non-abstaining answer.
- Require evidence for every fact or reasoning hop needed by the answer.

## Request options

- `max_iterations` defaults to `3` and must be positive.
- `max_tokens` controls answer generation only.
- `critic_max_tokens` controls critique generation, defaults to `4096`, and
  must be positive.

## Follow-up retrieval

Use at most three concrete missing-evidence issues from the Critic, bounded to
160 characters each. Use bounded feedback only when no usable issue exists.
The trace exposes accumulated and newly added document IDs for every round.

## Safe stopping

Stop with `Insufficient evidence to answer reliably.` when the iteration limit
is reached or a rejected round contributes no new evidence. Treat generated
abstentions as rejected even if a model Critic mistakenly approves them.
