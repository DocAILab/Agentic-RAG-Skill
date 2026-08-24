# SIM-RAG HotpotQA Optimization Design

## Goal

Improve the existing `agentic-sim-rag` inference loop after the 20-example
HotpotQA smoke test, without reproducing SIM-RAG training or changing existing
component contracts.

## Approved changes

1. Treat abstention-style candidates as insufficient even when the model Critic
   mistakenly approves them. The Critic prompt must require a direct answer and
   support for every necessary multi-hop fact.
2. Keep `max_tokens` for answer generation and add an independent
   `critic_max_tokens` request option, defaulting to 4096.
3. Build bounded follow-up retrieval queries from missing-evidence issues first;
   use truncated feedback only when no usable issue is returned.
4. Ask the grounded Generator for the shortest direct answer span so HotpotQA EM
   and token F1 are not reduced by unnecessary explanation.
5. Report `Recall@10` and `All-Support@10`, and expose per-iteration document IDs
   so multi-hop evidence gains can be inspected.

## Safety and compatibility

- Existing component input/output schemas remain unchanged.
- Existing `query`, `top_k`, `max_iterations`, and `max_tokens` behavior remains
  compatible; the new request option is optional.
- The workflow still returns the fixed safe answer when evidence never becomes
  sufficient.
- No external API calls or model downloads are part of implementation tests.

## Verification

Each behavior is introduced with a failing test, followed by the smallest code
change. Final checks cover the focused tests, full `pytest`, Ruff on changed
Python files, and Skill metadata validation. A paid HotpotQA rerun is prepared
but requires explicit confirmation before external API usage.
