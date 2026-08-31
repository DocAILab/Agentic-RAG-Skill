from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.financebench.scripts.build_demo import (
    REPOSITORY,
    REVISION,
    build_demo,
)
from experiments.retrieval.loading import (
    FINANCEBENCH_REVISION,
    iter_demo_items,
    iter_huggingface_items,
)
from experiments.retrieval.run_benchmark import (
    build_parser,
    build_run_metadata,
    main as run_benchmark_main,
)


def _row(
    identity: str,
    page: int,
    text: str,
    *,
    company: str = "Acme",
    doc_name: str = "ACME_2023_10K",
) -> dict:
    return {
        "financebench_id": identity,
        "company": company,
        "doc_name": doc_name,
        "question_type": "metrics-generated",
        "question": f"Question {identity}?",
        "answer": f"Answer {identity}",
        "evidence": [
            {
                "doc_name": doc_name,
                "evidence_page_num": page,
                "evidence_text": text,
                "evidence_text_full_page": text,
            }
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(root: Path) -> Path:
    source_root = root / "small"
    source_root.mkdir()
    source = source_root / "train.jsonl"
    rows = [
        _row(f"fb-{index}", index, f"Financial evidence page {index}.")
        for index in range(1, 13)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "name": "financebench-small-v1",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "split": "train",
            "sampling": {
                "method": "sha256-id-ranking",
                "salt": "FinanceBench-small-v1",
                "size": 12,
            },
        },
        "files": [
            {
                "path": "train.jsonl",
                "records": 12,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        ],
    }
    (source_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return source


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_build_demo_creates_shared_candidates_with_negatives(tmp_path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "demo"

    manifest = build_demo(source, output, limit=2, seed=17)
    corpus = _read_jsonl(output / "corpus.jsonl")
    tests = _read_jsonl(output / "test.jsonl")

    assert manifest["source"]["revision"] == REVISION
    assert manifest["source"]["license"] == "CC BY-NC 4.0"
    assert manifest["corpus"]["scope"] == "closed-evidence-pages"
    assert manifest["name"] == "financebench-evidence-page-demo-v2"
    assert manifest["candidates"]["size"] == 10
    assert len(corpus) == 12
    assert len(tests) == 2
    for example in tests:
        candidates = set(example["candidate_document_ids"])
        relevant = set(example["relevant_document_ids"])
        assert len(candidates) == 10
        assert candidates <= {record["id"] for record in corpus}
        assert relevant < candidates


def test_build_demo_is_reproducible_and_loader_restores_examples(tmp_path) -> None:
    source = _write_source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_demo(source, first, limit=2, seed=23)
    second_manifest = build_demo(source, second, limit=2, seed=23)
    items = list(iter_demo_items(first))

    assert first_manifest["files"] == second_manifest["files"]
    assert [item.sample_id for item in items] == [
        item.sample_id for item in iter_demo_items(second)
    ]
    assert all(item.error is None for item in items)
    assert all(item.example.label_type == "evidence_page" for item in items)
    assert all(len(item.example.documents) == 10 for item in items)


def test_candidate_pool_prioritizes_same_document_then_company() -> None:
    from experiments.financebench.scripts.build_demo import build_records

    rows = [
        _row("gold", 1, "Gold.", company="Acme", doc_name="ACME_2023_10K"),
        _row("same-doc", 2, "Same doc.", company="Acme", doc_name="ACME_2023_10K"),
        _row(
            "same-company",
            1,
            "Same company.",
            company="Acme",
            doc_name="ACME_2022_10K",
        ),
        _row("other", 1, "Other.", company="Other", doc_name="OTHER_2023_10K"),
    ]

    _, tests = build_records(
        rows,
        test_rows=[rows[0]],
        seed=7,
        candidate_count=3,
    )

    assert set(tests[0]["candidate_document_ids"]) == {
        "ACME_2023_10K#p1",
        "ACME_2023_10K#p2",
        "ACME_2022_10K#p1",
    }


def test_build_demo_rejects_source_that_does_not_match_manifest(tmp_path) -> None:
    source = _write_source(tmp_path)
    source.write_text(source.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        build_demo(source, tmp_path / "demo")


def test_financebench_huggingface_loader_pins_revision_and_rejects_fake_split() -> None:
    captured = {}

    def loader(path, **kwargs):
        captured.update(path=path, **kwargs)
        return iter([_row("fb-1", 1, "Revenue increased.")])

    item = next(
        iter_huggingface_items("financebench", "train", load_dataset_fn=loader)
    )

    assert item.error is None
    assert captured["revision"] == FINANCEBENCH_REVISION == REVISION
    with pytest.raises(ValueError, match="only provides split 'train'"):
        list(
            iter_huggingface_items(
                "financebench",
                "validation",
                load_dataset_fn=loader,
            )
        )


def test_financebench_cli_uses_demo_identity_and_test_split(tmp_path) -> None:
    source = _write_source(tmp_path)
    demo = tmp_path / "demo"
    build_demo(source, demo)
    args = build_parser().parse_args(
        [
            "--dataset",
            "financebench",
            "--retriever",
            "bm25",
            "--data-dir",
            str(demo),
        ]
    )

    metadata = build_run_metadata(args, code_commit="test")

    assert metadata["split"] == "test"
    assert metadata["data_dir"] == str(demo)
    assert metadata["dataset_manifest_sha256"] == _sha256(
        demo / "manifest.json"
    )


def test_financebench_demo_config_targets_generated_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "financebench" / "configs" / "demo.example.yaml"

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    demo = config["demo"]

    assert Path(demo["corpus_path"]).name == "corpus.jsonl"
    assert Path(demo["test_path"]).name == "test.jsonl"
    assert demo["candidate_documents_only"] is True
    assert demo["select_skills_per_example"] is False
    assert demo["request"]["constraints"]["retriever"] == (
        "component-bm25-retriever"
    )


def test_financebench_bm25_cli_runs_against_local_demo(tmp_path, capsys) -> None:
    source = _write_source(tmp_path)
    demo = tmp_path / "demo"
    output = tmp_path / "output"
    build_demo(source, demo)

    result = run_benchmark_main(
        [
            "--dataset",
            "financebench",
            "--retriever",
            "bm25",
            "--data-dir",
            str(demo),
            "--output-dir",
            str(output),
            "--top-k",
            "2",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert result == 0
    assert summary["counts"] == {
        "processed": 12,
        "ok": 12,
        "labelled": 12,
    }
    assert summary["metrics_by_label_type"]["evidence_page"]["count"] == 12
    assert (output / "results.jsonl").is_file()
