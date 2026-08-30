from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "TriviaQA"
    sys.modules.pop("triviaqa_data", None)
    sys.modules.pop("_manifest", None)
    monkeypatch.syspath_prepend(str(data_dir))
    return importlib.import_module("triviaqa_data")


def _load_cli(monkeypatch):
    data_dir = Path(__file__).resolve().parents[1] / "data" / "TriviaQA"
    sys.modules.pop("triviaqa_data", None)
    sys.modules.pop("_manifest", None)
    monkeypatch.syspath_prepend(str(data_dir))
    script_path = data_dir / "加载脚本.py"
    spec = importlib.util.spec_from_file_location("triviaqa_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_qa(path: Path, records) -> None:
    path.write_text(
        json.dumps({"Data": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def _row(
    question_id: str,
    *,
    question: str = "What is the capital of France?",
    answer: str = "Paris",
    aliases: list[str] | None = None,
    entity: list[dict] | None = None,
    search: list[dict] | None = None,
) -> dict:
    return {
        "QuestionId": question_id,
        "Question": question,
        "Answer": {"Value": answer, "Aliases": aliases or []},
        "EntityPages": entity
        if entity is not None
        else [{"Filename": "Paris.txt", "Title": "Paris"}],
        "SearchResults": search or [],
    }


def _evidence_root(tmp_path: Path) -> Path:
    wikipedia = tmp_path / "evidence" / "wikipedia"
    wikipedia.mkdir(parents=True)
    (wikipedia / "Paris.txt").write_text(
        "Paris is the capital of France.",
        encoding="utf-8",
    )
    web = tmp_path / "evidence" / "web" / "1"
    web.mkdir(parents=True)
    (web / "1_1.txt").write_text("France is in Europe.", encoding="utf-8")
    return tmp_path / "evidence"


def test_load_qa_records_reads_json_and_gzip(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    plain = tmp_path / "plain.json"
    _write_qa(plain, [_row("q1"), _row("q2")])
    compressed = tmp_path / "plain.json.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write(plain.read_text(encoding="utf-8"))

    assert [row["QuestionId"] for row in module.load_qa_records(plain)] == [
        "q1",
        "q2",
    ]
    assert [row["QuestionId"] for row in module.load_qa_records(compressed)] == [
        "q1",
        "q2",
    ]


def test_collect_valid_samples_filters_and_deduplicates(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    missing_entity = _row("q-missing", entity=[{"Filename": "Missing.txt"}])
    no_answer = _row("q-no-answer", answer="  ")
    invalid_answer = _row("q-invalid-answer")
    invalid_answer["Answer"] = None
    records = [
        _row("q1"),
        "dirty",
        _row("  "),
        _row("q-empty-query", question="  "),
        invalid_answer,
        no_answer,
        missing_entity,
        _row("q1"),
    ]

    samples, filtered = module.collect_valid_samples(records, evidence)

    assert [sample["sample_id"] for sample in samples] == ["q1"]
    assert filtered["dirty_record"] == 1
    assert filtered["empty_question_id"] == 1
    assert filtered["empty_query"] == 1
    assert filtered["invalid_answer"] == 1
    assert filtered["empty_answer"] == 1
    assert filtered["no_documents"] == 1
    assert filtered["duplicate_id"] == 1


def test_build_sample_uses_wikipedia_title_and_answer_aliases(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    sample, reason = module.build_sample(
        _row("q1", aliases=["PARIS", "Lutèce"]),
        evidence,
    )

    assert reason is None
    assert sample["query"] == "What is the capital of France?"
    assert sample["golden_answer"] == "Paris"
    assert sample["golden_answers"] == ["Paris", "PARIS", "Lutèce"]
    assert sample["documents"][0]["id"] == "wikipedia/Paris.txt"
    assert sample["documents"][0]["title"] == "Paris"
    assert sample["documents"][0]["text"] == "Paris is the capital of France."
    assert sample["documents"][0]["source"] == "wikipedia"


def test_windows_illegal_filename_fallback(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    wikipedia = tmp_path / "evidence" / "wikipedia"
    wikipedia.mkdir(parents=True)
    (wikipedia / "A_B.txt").write_text("Text for a sanitized filename.", encoding="utf-8")
    sample, _ = module.build_sample(
        _row("q1", entity=[{"Filename": "A:B.txt", "Title": "A"}], search=[]),
        tmp_path / "evidence",
    )

    assert sample["documents"][0]["id"] == "wikipedia/A_B.txt"


def test_web_evidence_reads_plain_text_fallback(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    sample, _ = module.build_sample(
        _row(
            "q-web",
            entity=[],
            search=[{"Filename": "1/1_1.txt", "Title": "Unrelated"}],
        ),
        evidence,
    )

    assert sample["documents"][0]["id"] == "web/1_1.txt"
    assert sample["documents"][0]["source"] == "web"
    assert sample["documents"][0]["text"] == "France is in Europe."


def test_build_subsets_is_deterministic_and_nested(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    samples = [
        {"sample_id": f"id-{index}", "documents": []} for index in range(20)
    ]

    forward = module.build_subsets(samples, [5, 20], seed=20260828)
    reverse = module.build_subsets(
        list(reversed(samples)),
        [20, 5, 20],
        seed=20260828,
    )

    assert [size for size, _ in forward] == [5, 20]
    assert [size for size, _ in reverse] == [5, 20]
    small_ids = {sample["sample_id"] for sample in forward[0][1]}
    large_ids = {sample["sample_id"] for sample in forward[1][1]}
    assert small_ids.issubset(large_ids)
    assert large_ids == {sample["sample_id"] for sample in reverse[1][1]}


def test_prepare_subsets_materializes_manifest_and_files(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])

    manifest = module.prepare_subsets(
        qa_path,
        evidence,
        tmp_path / "outputs",
        [2, 4],
    )

    subset_2 = json.loads(
        (tmp_path / "outputs" / "wikipedia-dev_subset_2.json").read_text(
            encoding="utf-8"
        )
    )
    subset_4 = json.loads(
        (tmp_path / "outputs" / "wikipedia-dev_subset_4.json").read_text(
            encoding="utf-8"
        )
    )
    assert subset_2["schema_version"] == module.SCHEMA_VERSION
    assert subset_2["dataset"] == "TriviaQA"
    assert subset_2["sampling"]["seed"] == module.DEFAULT_SEED
    assert subset_2["counts"]["samples"] == 2
    assert subset_4["counts"]["samples"] == 4
    assert {
        sample["sample_id"] for sample in subset_2["samples"]
    }.issubset({sample["sample_id"] for sample in subset_4["samples"]})

    assert manifest["name"] == module.SUBSET_MANIFEST_NAME
    assert manifest["source"]["revision"] == module.sha256_file(qa_path)
    assert manifest["sampling"]["seed"] == module.DEFAULT_SEED
    assert [record["records"] for record in manifest["files"]] == [2, 4]
    assert manifest["files"][0]["sha256"] == module.sha256_file(
        tmp_path / "outputs" / "wikipedia-dev_subset_2.json"
    )


def test_prepare_subsets_reuses_valid_files_without_rereading(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])
    output_root = tmp_path / "outputs"

    first = module.prepare_subsets(qa_path, evidence, output_root, [2])

    def unexpected_loader(*args, **kwargs):
        raise AssertionError("valid subsets should not be rebuilt")

    second = module.prepare_subsets(
        qa_path,
        evidence,
        output_root,
        [2],
        load_records_fn=unexpected_loader,
    )

    assert second == first


def test_prepare_subsets_writes_only_missing_sizes(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])
    output_root = tmp_path / "outputs"

    module.prepare_subsets(qa_path, evidence, output_root, [2])
    kept_path = output_root / "wikipedia-dev_subset_2.json"
    kept_mtime = kept_path.stat().st_mtime_ns

    manifest = module.prepare_subsets(qa_path, evidence, output_root, [2, 4])

    assert [record["records"] for record in manifest["files"]] == [2, 4]
    assert kept_path.stat().st_mtime_ns == kept_mtime


def test_corrupt_subset_requires_force_before_rebuild(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])
    output_root = tmp_path / "outputs"
    module.prepare_subsets(qa_path, evidence, output_root, [2])
    subset_path = output_root / "wikipedia-dev_subset_2.json"
    with subset_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(module.DatasetStateError, match="rerun with --force"):
        module.prepare_subsets(qa_path, evidence, output_root, [2])

    manifest = module.prepare_subsets(
        qa_path,
        evidence,
        output_root,
        [2],
        force=True,
    )
    assert manifest["files"][0]["sha256"] == module.sha256_file(subset_path)


def test_seed_mismatch_requires_force_before_rebuild(
    tmp_path, monkeypatch
) -> None:
    module = _load_module(monkeypatch)
    evidence = _evidence_root(tmp_path)
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])
    output_root = tmp_path / "outputs"
    module.prepare_subsets(qa_path, evidence, output_root, [2], seed=1)

    with pytest.raises(module.DatasetStateError, match="--force"):
        module.prepare_subsets(qa_path, evidence, output_root, [2], seed=2)

    manifest = module.prepare_subsets(
        qa_path,
        evidence,
        output_root,
        [2],
        seed=2,
        force=True,
    )
    assert manifest["sampling"]["seed"] == 2


def test_changed_evidence_requires_force_before_rebuild(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    evidence_a = _evidence_root(tmp_path)
    evidence_b = tmp_path / "evidence-b" / "wikipedia"
    evidence_b.mkdir(parents=True)
    (evidence_b / "Paris.txt").write_text(
        "Paris is the largest city in France.",
        encoding="utf-8",
    )
    qa_path = tmp_path / "wikipedia-dev.json"
    _write_qa(qa_path, [_row(f"q{index}") for index in range(6)])
    output_root = tmp_path / "outputs"
    module.prepare_subsets(qa_path, evidence_a, output_root, [2])

    with pytest.raises(module.DatasetStateError, match="--force"):
        module.prepare_subsets(qa_path, tmp_path / "evidence-b", output_root, [2])

    manifest = module.prepare_subsets(
        qa_path,
        tmp_path / "evidence-b",
        output_root,
        [2],
        force=True,
    )
    subset = json.loads(
        (output_root / "wikipedia-dev_subset_2.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["evidence"] == str(tmp_path / "evidence-b")
    assert subset["samples"][0]["documents"][0]["text"] == (
        "Paris is the largest city in France."
    )


def test_missing_source_raises_before_writing(tmp_path, monkeypatch) -> None:
    module = _load_module(monkeypatch)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        module.prepare_subsets(
            tmp_path / "missing.json",
            tmp_path / "evidence",
            tmp_path / "outputs",
            [2],
        )


def test_cli_defaults(monkeypatch) -> None:
    module = _load_cli(monkeypatch)
    args = module.parse_args([])

    assert args.source == module.DEFAULT_SOURCE
    assert args.evidence == module.DEFAULT_EVIDENCE
    assert args.output == module.DEFAULT_OUTPUT
    assert args.max_query_samples == [100, 800]
    assert args.seed == 20260828
    assert args.force is False


def test_cli_main_prints_summary(tmp_path, monkeypatch, capsys) -> None:
    module = _load_cli(monkeypatch)
    manifest = {
        "name": "triviaqa-subsets-v1",
        "counts": {"samples": 5, "filtered": {"no_documents": 1}},
        "files": [{"path": "wikipedia-dev_subset_5.json", "records": 5, "bytes": 10}],
    }
    monkeypatch.setattr(module, "prepare_subsets", lambda **kwargs: manifest)

    result = module.main(
        [
            "--source",
            "qa.json",
            "--evidence",
            "evidence",
            "--output",
            str(tmp_path),
            "--max-query-samples",
            "5",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert result == 0
    assert summary["name"] == "triviaqa-subsets-v1"
    assert summary["files"] == [
        {"path": "wikipedia-dev_subset_5.json", "records": 5, "bytes": 10}
    ]
