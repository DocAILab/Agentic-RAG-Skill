# SIM-RAG Answer Regeneration Implementation Plan

## Scope

Implement the approved answer-regeneration branch in
`agentic-sim-rag` without changing any Component manifest or public Component
request/result schema.

## TDD slices

1. Add an integration-style workflow test for a format-only Critic rejection.
   Assert one retrieval, two Generator calls, two Critic calls, the revised
   answer, a regeneration trace event, and the new stop reason. Implement the
   smallest conservative rejection classifier and regeneration path.
2. Add a test that a clear evidence-gap rejection performs another retrieval
   and never regenerates in that round. Preserve the current follow-up query.
3. Add a mixed-reason test. Default to retrieval when answer-form and
   evidence-gap signals coexist.
4. Add a test for a revised answer that is still rejected. Ensure it is not
   returned and existing maximum-iteration/no-new-evidence safety applies.
5. Refactor bounded revision guidance and trace construction into short helper
   functions. Update the Agentic Skill contract and manifest version.

## Verification

- Run focused SIM-RAG tests after every RED-GREEN slice.
- Run all framework tests.
- Run Ruff on every changed Python file and `git diff --check`.
- Do not run a new HotpotQA experiment until the user reviews the code changes.

