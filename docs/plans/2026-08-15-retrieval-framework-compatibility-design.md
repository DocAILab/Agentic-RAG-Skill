# Retrieval framework compatibility test design

## Goal

Verify that the updated BM25F and BGE V2 retrieval components can still be
discovered, selected, bound, called, and executed by the original framework.
This test covers interface and execution compatibility. It does not repeat the
retrieval-quality benchmark or evaluate answer generation quality.

## Test path

Use one original RRFusion execution path:

```text
Manage selection
  -> RRFusion Agentic Skill
  -> BM25F Retriever
  -> BGE V2 Retriever
  -> reciprocal-rank fusion
  -> existing Grounded Generator
```

The test uses scripted selection responses and a deterministic fake embedding
service. It therefore requires no model download, API key, network call, or
generation cost.

## Assertions

1. The framework discovers the existing RRFusion Agentic Skill and both updated
   Retriever Components.
2. Component selection binds RRFusion, BM25F, BGE V2, and the existing Grounded
   Generator without special-case code.
3. Both retrievers receive the unchanged `RetrievalRequest` shape and return the
   unchanged `RetrievalResult` shape.
4. RRFusion combines the two ranked document lists and passes the fused evidence
   to the existing generator.
5. The final result contains a non-empty answer, documents, selection records,
   and a `retrieve/fuse -> generate` trace.

## Scope limits

- Do not run full HotpotQA, 2WikiMultihopQA, or TriviaQA experiments.
- Do not ask the selection model to retune retrieval parameters.
- Do not compare Hit, Recall, AllSupport, MRR, EM, or F1.
- Do not modify Manage, Agentic, Generator, or public request/result interfaces.
- Treat this as integration evidence only, not retrieval-quality evidence.
