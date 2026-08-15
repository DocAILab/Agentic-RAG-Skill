# Retrieval component experiment design

Status: approved on 2026-08-15

## 1. Objective and scope

Evaluate only the Retrieval work assigned in phase one. The experiment must:

1. compare the current BM25F and BGE-Large retrievers with the original
   framework implementations;
2. attribute gains to the main retrieval changes through controlled ablations;
3. select one cross-dataset default BM25F configuration and one default BGE
   document representation;
4. determine the smallest useful `top_k` for downstream context efficiency.

The benchmark directly calls a retriever for each question's candidate
documents. It does not call Manage, Agentic, Reranker, Generator, DeepSeek, a
global index, or an external search service.

## 2. Research questions and hypotheses

### RQ1: Does field-aware BM25F improve over the original text-only BM25?

- H0: BM25F has no practically meaningful improvement in macro AllSupport@5.
- H1: BM25F improves macro AllSupport@5 because titles contain entity signals
  that should not be diluted by body length.
- Falsification: reject H1 when the paired 95% confidence interval includes
  zero, the absolute improvement is below one percentage point, or both
  strong-label datasets regress.

### RQ2: Does the current BGE usage improve over the original Vector retriever?

- H0: the official query instruction and title-plus-text passage have no
  practically meaningful retrieval benefit.
- H1: the instruction improves query representation and the title supplies
  entity evidence, increasing MRR and AllSupport@5.
- Falsification: reject H1 under the same confidence and practical-effect rules
  used for RQ1.

### RQ3: Can the improved retrievers use fewer documents?

- H0: reducing `top_k` substantially lowers support recall.
- H1: an improved retriever reaches at least 95% of its Top-10 AllSupport and
  Recall at a smaller K.

## 3. Experimental design decision

Use a two-stage frozen-configuration design.

1. Tune and screen on deterministic subsets drawn only from training data.
2. Freeze every choice before inspecting full validation results.
3. Run the frozen systems on complete labelled validation/dev splits.
4. Report TriviaQA weak-label results separately and never use them to choose
   defaults.

This design is preferred over full-split grid search because it avoids tuning
on the final evaluation data. It is preferred over a single default run because
the latter cannot justify the defaults or isolate improvement sources.

## 4. Systems under comparison

### BM25 family

| ID | System | Purpose |
| --- | --- | --- |
| B0 | Original text-only Okapi BM25 | Initial-framework baseline |
| B1 | Title-weighted BM25 with combined length | Isolate adding title evidence |
| B2 | Current BM25F defaults | Isolate field-aware normalization |
| B3 | Frozen tuned BM25F | Select final defaults |

B0 reproduces baseline commit `5b9197a`. B1 reproduces the intermediate
implementation before commit `999b63e`. B2 uses `k1=1.5`, `b=0.75`,
`title_b=0.75`, and `title_boost=1.5`.

### Vector family

All Vector systems use `BAAI/bge-large-en-v1.5`, the same embedding client,
device, batch size, normalized vectors, and cosine ranking.

| ID | Query representation | Document representation | Purpose |
| --- | --- | --- | --- |
| V0 | Raw query | Text only | Initial-framework baseline |
| V1 | Official query instruction | Text only | Isolate instruction effect |
| V2 | Official query instruction | Title plus text | Current implementation |

Precomputed embeddings, batching, and load-once behavior are efficiency
features. They receive separate invariance and throughput checks rather than a
retrieval-quality hypothesis.

## 5. Data and sampling

### Tuning and screening data

- HotpotQA train: 10,000 examples for BM25F tuning; 2,000 for BGE screening.
- 2WikiMultihopQA train: 10,000 examples for BM25F tuning; 2,000 for BGE
  screening.
- TriviaQA: excluded from tuning because its relevance labels are weak.

Choose samples by sorting the SHA-256 digest of `dataset + ":" + sample_id` and
taking the lowest N values. Persist the selected IDs so every configuration
uses exactly the same examples.

### Frozen final evaluation data

- HotpotQA `validation`, using strong `supporting_facts` labels.
- 2WikiMultihopQA `dev`, exposed by the CLI alias `validation`, using strong
  supporting-fact labels.
- TriviaQA `validation`, using `weak_answer_alias` labels and a separate report.

Unlabelled test splits may save rankings but must not produce pseudo-metrics.

## 6. BM25F parameter search

Evaluate this predeclared 72-configuration grid:

```text
k1          = [1.2, 1.5, 2.0]
b           = [0.5, 0.75]
title_b     = [0.0, 0.5, 0.75]
title_boost = [1.0, 1.5, 2.0, 3.0]
```

Select the frozen default with these rules, in order:

1. maximize the macro mean of HotpotQA and 2Wiki AllSupport@5;
2. use macro Recall@5 and MRR as tie-breakers;
3. reject configurations that reduce either dataset's Recall@10 by more than
   0.5 percentage points relative to B2;
4. when macro AllSupport@5 differs by no more than 0.2 percentage points,
   select the configuration closest to the conventional current defaults;
5. select one global default, not dataset-specific defaults.

Full validation results cannot be used to retune the grid.

## 7. BGE strategy screening

Run V0, V1, and V2 on both fixed 2,000-example screening subsets. Rank them by
the same strong-label objective used for BM25F. Freeze the best improved
strategy, then run only V0 and that strategy on the complete validation/dev
splits. Keep the official instruction and vector normalization fixed rather
than treating known model-use requirements as free hyperparameters.

Choose `batch_size` separately as the largest tested value in `[8, 16, 32, 64]`
that does not cause out-of-memory errors on the target device. Report median
throughput and peak memory; do not interpret batch size as a quality parameter.

## 8. Metrics and default Top-K

Compute each retrieval metric at `K = 1, 2, 3, 5, 10`:

- Hit@K;
- Recall@K;
- AllSupport@K;
- MRR.

Primary metric: macro AllSupport@5 across HotpotQA and 2Wiki.

Secondary metrics: Recall@5, MRR, Hit@1, AllSupport@10, latency, throughput,
peak memory, returned document count, and returned evidence-token count.

For each frozen retriever, choose the smallest K satisfying both:

```text
AllSupport@K >= 0.95 * AllSupport@10
Recall@K     >= 0.95 * Recall@10
```

This K is the recommended downstream default. The benchmark still retrieves
Top-10 so all cutoffs can be computed from one ranking.

## 9. Statistical analysis

Comparisons are paired because every system processes the same examples.

- Report per-dataset means and absolute percentage-point differences.
- Report the macro mean only for the two strong-label datasets.
- Estimate paired 95% confidence intervals with 10,000 bootstrap resamples and
  a fixed seed of `20260815`.
- Prefer effect sizes and confidence intervals to binary significance claims.
- Treat a one-percentage-point AllSupport@5 gain as the minimum practical
  improvement for a superiority claim.
- If p-values are added for multiple pairwise comparisons, apply Holm
  correction; p-values are not required for default selection.

## 10. Error analysis

For each important pair, save up to 50 deterministic examples from each
direction of disagreement. Classify them as:

- title-entity match;
- body lexical match;
- paraphrase or weak lexical overlap;
- rare entity;
- one supporting document found;
- no supporting document found;
- suspected weak-label error.

The categories explain mechanisms but do not replace the preregistered metrics.

## 11. Outputs

```text
experiments/retrieval/outputs/
├── tuning/
│   ├── manifests/
│   ├── bm25f_grid.jsonl
│   └── selected_defaults.json
├── screening/
│   └── bge_variants/
├── full/
│   ├── hotpotqa/
│   ├── 2wiki/
│   └── triviaqa/
├── comparisons/
│   ├── paired_bootstrap.json
│   ├── results_table.csv
│   └── failure_cases.jsonl
└── final_retrieval_config.json
```

Every result records dataset, split, sample manifest, retriever variant,
algorithm parameters, model, representation policy, Top-K, code commit, and run
signature. Resume is allowed only when the full signature matches.

## 12. Validity limits

- The benchmark evaluates per-question candidate-document ranking, not global
  corpus indexing or approximate-nearest-neighbour search.
- Retrieval metrics cannot establish final answer quality.
- TriviaQA labels are weak and cannot be pooled with supporting-fact labels.
- BGE results depend on the fixed English model and English datasets.
- Full validation is a confirmation set; changing parameters afterward voids
  the confirmatory interpretation.

## 13. Acceptance criteria

The experiment is complete when:

1. all baseline and ablation variants are reproducible from saved configs;
2. the tuning manifest and frozen choices are persisted before full evaluation;
3. all three labelled validation/dev evaluations finish or record explicit
   unrecoverable samples;
4. paired confidence intervals and practical effect sizes are reported;
5. final BM25F parameters, BGE representation, and Top-K are justified by the
   predeclared rules;
6. no DeepSeek or generation call occurs.
