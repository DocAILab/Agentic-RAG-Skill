# Retrieval experiment implementation plan

Depends on: `docs/plans/2026-08-15-retrieval-experiment-design.md`

This plan implements the approved experiment without changing the phase-one
scope or the public `RetrievalRequest -> RetrievalResult` contract.

## Task 1: Represent experiment variants and complete run signatures

Files:

- modify `experiments/retrieval/retrievers.py`;
- modify `experiments/retrieval/run_benchmark.py`;
- modify `experiments/retrieval/persistence.py`;
- add or extend benchmark configuration tests.

Work:

1. Add explicit retriever-variant names B0-B3 and V0-V2.
2. Expose `k1`, `b`, `title_b`, and `title_boost` for BM25F runs.
3. Expose the experimental query-instruction and document-field policies needed
   for V0-V2 without changing default Component behavior.
4. Include all ranking-affecting options, sample-manifest identity, and code
   commit in metadata and the resume signature.
5. Keep `device`, `batch_size`, and checkpoint frequency out of quality identity
   unless they can change produced rankings.

Verification:

- tests reject resume when any quality parameter or manifest differs;
- existing CLI commands keep their current behavior;
- default component requests remain backward compatible.

## Task 2: Add reproducible original baselines

Files:

- add `experiments/retrieval/baselines.py`;
- modify `experiments/retrieval/retrievers.py`;
- add `tests/test_retrieval_baselines.py`.

Work:

1. Reproduce B0 from commit `5b9197a` as an experiment-only baseline.
2. Reproduce B1 from the pre-BM25F implementation as an experiment-only
   baseline.
3. Build V0 and V1 by controlling query instruction and projected document
   fields while reusing the same embedding client as V2.
4. Do not restore weaker baseline behavior inside production Components.

Verification:

- golden fixtures reproduce known original rankings;
- V0-V2 use identical model instances and differ only in declared
  representations;
- cached and uncached embeddings give identical rankings.

## Task 3: Add deterministic tuning manifests

Files:

- add `experiments/retrieval/sampling.py`;
- add a manifest CLI or subcommand;
- add `tests/test_retrieval_sampling.py`.

Work:

1. Select examples by SHA-256 of `dataset + ":" + sample_id`.
2. Persist dataset, split, requested size, selected IDs, and digest.
3. Let every tuning/screening run consume the same manifest.
4. Fail when the requested manifest cannot be satisfied.

Verification:

- input order does not change selected IDs;
- repeated generation is byte-for-byte stable;
- HotpotQA and 2Wiki manifests cannot be accidentally interchanged.

## Task 4: Extend metrics to the approved cutoffs

Files:

- modify `experiments/retrieval/scoring.py`;
- extend `tests/test_retrieval_metrics.py`.

Work:

1. Change evaluated ranks to `1, 2, 3, 5, 10`.
2. Record returned document count and evidence-token count with one documented
   tokenizer policy.
3. Keep strong and weak label summaries separate.
4. Add a function that selects the smallest K meeting both 95% retention rules.

Verification:

- hand-computed multi-hop fixtures cover partial and complete support;
- no-label examples still produce `metrics = null`;
- Top-K selection handles zero-denominator and tie cases explicitly.

## Task 5: Implement BM25F grid search and freezing

Files:

- add `experiments/retrieval/run_bm25f_tuning.py`;
- add `experiments/retrieval/selection.py`;
- add `tests/test_retrieval_parameter_selection.py`.

Work:

1. Enumerate exactly the 72 approved configurations.
2. Run both strong-label tuning manifests for every configuration.
3. Append one record per configuration and resume completed configurations.
4. Apply the ordered selection rules without manual intervention.
5. Write `selected_defaults.json` and mark it frozen.

Verification:

- the grid contains 72 unique configurations including current defaults;
- synthetic summaries verify every tie-break and rejection rule;
- a frozen file cannot be silently overwritten with different inputs.

## Task 6: Implement BGE screening and efficiency measurements

Files:

- add `experiments/retrieval/run_bge_screening.py`;
- add `experiments/retrieval/performance.py`;
- add focused tests with a fake embedding client.

Work:

1. Run V0-V2 on the two fixed 2,000-example manifests.
2. Freeze the best improved representation using the approved objective.
3. Benchmark batch sizes `[8, 16, 32, 64]` after warm-up.
4. Record throughput and peak memory when the runtime exposes it.
5. Treat out-of-memory as a rejected efficiency setting, not a failed quality
   experiment.

Verification:

- fake embeddings prove the variants construct the expected texts;
- ranking-affecting and efficiency-only settings remain separated;
- the real-model integration test stays opt-in.

## Task 7: Add paired statistical comparison

Files:

- add `experiments/retrieval/analysis.py`;
- add `experiments/retrieval/run_analysis.py`;
- add `tests/test_retrieval_analysis.py`.

Work:

1. Align systems by dataset, sample ID, and manifest.
2. Reject comparisons with missing or duplicate paired examples.
3. Compute absolute paired differences and 10,000 bootstrap intervals using
   seed `20260815`.
4. Compute strong-label macro summaries without pooling TriviaQA.
5. Apply the preregistered practical-effect and falsification rules.

Verification:

- deterministic fixtures reproduce exact intervals for a fixed seed;
- identical systems yield zero differences;
- deliberately misaligned results are rejected.

## Task 8: Generate final tables and failure sets

Files:

- add `experiments/retrieval/reporting.py`;
- extend `experiments/retrieval/README.md`;
- add report-output tests.

Work:

1. Write machine-readable comparison JSON and a flat CSV table.
2. Select deterministic directional disagreements for error analysis.
3. Preserve labels and retrieval ranks without copying full embedding vectors.
4. Write `final_retrieval_config.json` containing BM25F, BGE representation,
   and recommended Top-K.

Verification:

- outputs are deterministic and valid UTF-8;
- weak-label rows are visibly separated;
- the final config traces back to frozen tuning and screening records.

## Task 9: Verification ladder before expensive runs

Run in this order:

1. unit tests for configuration, baselines, sampling, metrics, selection, and
   analysis;
2. complete local test suite and Ruff;
3. one-example smoke tests for B0-B3 and V0-V2 with fake embeddings;
4. ten-example real BM25 and real BGE smoke tests on every dataset adapter;
5. generate and freeze tuning/screening manifests;
6. run BM25F tuning and freeze B3;
7. run BGE screening and freeze the improved Vector strategy;
8. run complete validation/dev evaluations;
9. run paired analysis, create final config, and inspect failure samples.

Do not start the complete BGE evaluation until all earlier checks pass. Do not
change frozen defaults after inspecting complete validation results; any later
change starts a new explicitly versioned experiment.
