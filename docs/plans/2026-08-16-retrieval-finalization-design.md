# Retrieval branch finalization design

## Goal

Prepare `feature/ywq-bm25-vector` for review without committing large benchmark
artifacts. The reviewed code, its defaults, and the compact experiment evidence
must agree.

## Decision

Use a compact, reviewable finalization:

1. Make the selected BM25F B3 parameters the component and benchmark defaults.
2. Keep BGE V2 as the default representation and use the measured best batch
   size of 8 in the benchmark entry point.
3. Add a tracked `experiments/retrieval/RESULTS.md` containing dataset sizes,
   paired-bootstrap effects, selected defaults, limitations, and provenance.
4. Keep raw JSONL, checkpoints, summaries, and model artifacts ignored.
5. Regenerate the ignored aggregate result table locally so the workspace stays
   internally consistent.
6. Ignore IDE metadata and fix the existing whitespace warning.

## Alternatives considered

### Commit all experiment outputs

Rejected because full JSONL files are large, derived, and reproducible from the
tracked workflow. They would make review and repository maintenance harder.

### Leave defaults and results outside version control

Rejected because reviewers would receive working experiment code but not the
validated configuration or the evidence used to select it.

## Compatibility

The public `run(inputs, context)` interface and request/result schemas remain
unchanged. Callers can still override every retrieval parameter. Only omitted
parameter behavior changes to the experimentally selected defaults.

## Verification

- Add or update tests that assert default BM25F parameters and default variants.
- Run the complete pytest suite.
- Run `git diff --check`.
- Confirm benchmark outputs remain ignored and `.idea/` is ignored.
- Confirm the aggregate table includes all six full paired comparisons.
