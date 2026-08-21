# HotpotQA Experiment Data

## Source

- Dataset: HotpotQA
- Official Hugging Face repository: `hotpotqa/hotpot_qa`
- Original project: `https://hotpotqa.github.io/`
- License: CC BY-SA 4.0
- Retrieved: 2026-07-29

The local snapshot preserves the official Hugging Face `distractor` and
`fullwiki` configuration layout. Their train shards are byte-identical, so the
`fullwiki/train-*.parquet` files are NTFS hard links to the corresponding
`distractor` files.

## Local layout

```text
data/raw/
|-- distractor/
|   |-- train-00000-of-00002.parquet
|   |-- train-00001-of-00002.parquet
|   `-- validation-00000-of-00001.parquet
`-- fullwiki/
    |-- train-00000-of-00002.parquet
    |-- train-00001-of-00002.parquet
    |-- validation-00000-of-00001.parquet
    `-- test-00000-of-00001.parquet
```

Raw data is immutable input. Derived datasets, indexes, caches, predictions,
and evaluation results must be written outside `data/raw/`.

## Demo subset

`data/demo/` contains a small, tracked subset derived from the distractor
validation split for framework demonstrations:

- `corpus.jsonl`: 2,000 shared candidate documents with stable title IDs.
- `test.jsonl`: 100 questions with answers, relevant document IDs, original
  ten-document candidate sets, and sentence-level supporting facts.
- `manifest.json`: deterministic sampling configuration, counts, source hash,
  and output hashes.
- `README.md`: data contract and framework usage example.

The test set uses seed `20260807` and balances 50 bridge with 50 comparison
questions. Additional validation contexts deterministically expand the shared
corpus to 2,000 unique documents. Rebuild it without modifying raw data:

```powershell
python -B experiments/hotpotqa/scripts/build_demo.py
```
