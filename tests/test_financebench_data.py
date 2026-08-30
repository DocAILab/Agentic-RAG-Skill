from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def _load_module(monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "FinanceBench"
    sys.modules.pop("financebench_data", None)
    sys.modules.pop("_manifest", None)
    monkeypatch.syspath_prepend(str(data_dir))
    return importlib.import_module("financebench_data")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _row(financebench_id: str, doc_name: str, page: int, text: str) -> dict:
    return {
        "financebench_id": financebench_id,
        "company": "Acme",
        "doc_name": doc_name,
        "question_type": "domain-relevant",
        "question": f"What about {financebench_id}?",
        "answer": "Widgets",
        "evidence": [
            {
                "doc_name": doc_name,
                "evidence_page_num": page,
                "evidence_text": text,
                "evidence_text_full_page": text,
            }
        ],
    }


def test_prepare_versions_materializes_public_and_small_deterministically(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    rows = [
        _row("fb-1", "ACME_2023_10K", 12, "Acme makes widgets."),
        _row("fb-2", "ACME_2023_10K", 18, "Acme also sells services."),
        _row("fb-3", "ACME_2024_10K", 3, "Acme reports a new segment."),
    ]
    calls = []

    def loader(dataset_name, *, split, revision, streaming):
        calls.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "revision": revision,
                "streaming": streaming,
            }
        )
        return iter(rows)

    manifests = module.prepare_versions(
        "all",
        tmp_path,
        load_dataset_fn=loader,
        small_size=2,
    )

    assert calls == [
        {
            "dataset_name": module.REPOSITORY,
            "split": module.DEFAULT_SPLIT,
            "revision": module.REVISION,
            "streaming": True,
        }
    ]
    assert set(manifests) == {"public", "small"}
    assert manifests["public"]["name"] == module.PUBLIC_NAME
    assert manifests["small"]["name"] == module.SMALL_NAME
    assert manifests["public"]["counts"]["examples"] == 3
    assert manifests["small"]["counts"]["examples"] == 2

    public_rows = _read_jsonl(tmp_path / "public" / "train.jsonl")
    small_rows = _read_jsonl(tmp_path / "small" / "train.jsonl")
    assert [row["financebench_id"] for row in public_rows] == [
        "fb-1",
        "fb-2",
        "fb-3",
    ]
    assert len(small_rows) == 2
    assert {
        row["financebench_id"] for row in small_rows
    }.issubset({row["financebench_id"] for row in public_rows})

    reverse_root = tmp_path / "reverse"

    module.prepare_versions(
        "small",
        reverse_root,
        load_dataset_fn=lambda *args, **kwargs: iter(reversed(rows)),
        small_size=2,
    )

    reverse_small_rows = _read_jsonl(reverse_root / "small" / "train.jsonl")
    assert [row["financebench_id"] for row in small_rows] == [
        row["financebench_id"] for row in reverse_small_rows
    ]
