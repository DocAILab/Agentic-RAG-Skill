# Retrieval comparison experiments

This workflow evaluates only candidate-document retrieval. It never calls the
Manage, Agentic, Reranker, Generator, or DeepSeek layers.

The completed experiment results and selected defaults are recorded in
[RESULTS.md](RESULTS.md).

## 1. Create frozen training manifests

```powershell
python -m experiments.retrieval.run_manifest `
  --dataset hotpotqa --size 10000 `
  --output experiments/retrieval/outputs/tuning/manifests/hotpot-10000.json

python -m experiments.retrieval.run_manifest `
  --dataset 2wiki --size 10000 `
  --output experiments/retrieval/outputs/tuning/manifests/2wiki-10000.json
```

Create separate 2,000-example manifests for BGE screening.

## 2. Tune and freeze BM25F

```powershell
python -m experiments.retrieval.run_bm25f_tuning `
  --hotpot-manifest experiments/retrieval/outputs/tuning/manifests/hotpot-10000.json `
  --two-wiki-manifest experiments/retrieval/outputs/tuning/manifests/2wiki-10000.json `
  --output-dir experiments/retrieval/outputs/tuning
```

The runner evaluates the predeclared 72 configurations, resumes completed
configurations, and writes `selected_defaults.json` without overwriting a
different frozen choice.

## 3. Screen BGE variants

```powershell
python -m experiments.retrieval.run_bge_screening `
  --hotpot-manifest experiments/retrieval/outputs/tuning/manifests/hotpot-2000.json `
  --two-wiki-manifest experiments/retrieval/outputs/tuning/manifests/2wiki-2000.json `
  --output-dir experiments/retrieval/outputs/screening/bge `
  --model BAAI/bge-large-en-v1.5 --device cuda --batch-size 8
```

Variants are V0 raw-query/text-only, V1 instructed-query/text-only, and V2
instructed-query/title-plus-text.

## 4. Run frozen full comparisons

Use `--variant B0`, `B1`, `B2`, or `B3` with `--retriever bm25`. The command
defaults to the frozen B3 values; pass `--k1`, `--b`, `--title-b`, and
`--title-boost` to reproduce another variant explicitly.

Use `--variant V0`, `V1`, or `V2` with `--retriever vector`. Full evaluation
should compare V0 only with the improved variant frozen during screening.

Every ranking-affecting option, manifest digest, and code commit is part of the
resume signature. Batch size and checkpoint frequency are efficiency settings.

## 5. Compare paired runs

```powershell
python -m experiments.retrieval.run_analysis `
  --dataset hotpotqa `
  --baseline experiments/retrieval/outputs/full/hotpotqa/b0/results.jsonl `
  --candidate experiments/retrieval/outputs/full/hotpotqa/b3/results.jsonl `
  --output experiments/retrieval/outputs/comparisons/hotpot-b0-b3.json
```

The analysis requires identical labelled sample IDs and reports candidate minus
baseline differences with deterministic paired bootstrap confidence intervals.
