from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from experiments.retrieval.adapters import adapt_hotpotqa


PROJECT_ROOT = Path(__file__).parents[1]
DATA_MODULE_ROOT = PROJECT_ROOT / "data" / "HotpotQA"
sys.path.insert(0, str(DATA_MODULE_ROOT))
hotpotqa_data = importlib.import_module("hotpotqa_data")


def _source_dataset(count: int) -> Dataset:
    rows = [
        {
            "id": f"integration-{index}",
            "question": f"Which document supports answer {index}?",
            "answer": f"Answer {index}",
            "type": "bridge" if index % 2 else "comparison",
            "level": "hard",
            "supporting_facts": {"title": [f"Gold {index}"], "sent_id": [0]},
            "context": {
                "title": [f"Gold {index}", f"Distractor {index}"],
                "sentences": [["Supporting evidence."], ["Other evidence."]],
            },
        }
        for index in range(count)
    ]
    return Dataset.from_list(rows)


def test_all_5000_small_rows_pass_the_shared_hotpotqa_adapter(tmp_path) -> None:
    source = _source_dataset(5_001)
    hotpotqa_data.prepare_small(
        tmp_path,
        load_dataset_fn=lambda *args, **kwargs: source,
    )
    rows = pq.read_table(tmp_path / "small" / "train-5000.parquet").to_pylist()

    examples = [adapt_hotpotqa(row) for row in rows]

    assert len(examples) == 5_000
    assert all(len(example.documents) == 2 for example in examples)
    assert all(len(example.relevant_document_ids) == 1 for example in examples)
    assert all(example.label_type == "supporting_facts" for example in examples)
