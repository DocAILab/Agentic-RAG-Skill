---
name: manage-rag-default
description: Choose exactly one Agentic RAG workflow for a query using task complexity, evidence needs, corpus characteristics, and execution budget. Use as the first and only Skill loaded during the Manage selection stage.
---

# Default RAG Manager

Analyze the query and execution state before any Agentic or Component Skill body is loaded.

## Selection procedure

1. Determine whether the task needs retrieval.
2. Estimate whether one retrieval route is sufficient or complementary routes are needed.
3. Consider corpus size, lexical specificity, semantic mismatch, latency budget, and token budget.
4. Request advertisements for Agentic Skills only.
5. Produce guidance that will be used to select exactly one advertised Agentic Skill.

## Baseline guidance

- Prefer `agentic-vanilla-rag` for ordinary single-route retrieval followed by grounded generation.
- Prefer `agentic-rrfusion` when lexical and semantic retrieval are complementary or recall is more important than minimum latency.
- Reject any candidate that is not an Agentic Skill.
- Do not select Component Skills directly.

## Output

Return strict JSON without selecting a Component Skill:

```json
{"agentic_selection_guidance": "selection guidance", "reason": "short reason"}
```

The framework will expose Agentic advertisements in the next stage and perform the actual selection.
