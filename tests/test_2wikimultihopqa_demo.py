from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "2WikiMultihopQA"
MODULE_PATH = DATA_ROOT / "two_wiki_data.py"
SAMPLE_MANIFEST = DATA_ROOT / "demo" / "sample_manifest.json"
LOCAL_DEMO_ROOT = DATA_ROOT / "demo"


def _load_module():
    spec = importlib.util.spec_from_file_location("two_wiki_data", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


two_wiki = _load_module()


def _row(
    identity: str,
    *,
    context=None,
    supporting_facts=None,
    evidences=None,
    answer: str = "Answer",
    question_type: str = "compositional",
) -> dict:
    context = context or [
        ["Alpha", ["Alpha sentence zero.", "Alpha sentence one."]],
        ["Beta", ["Beta sentence zero.", "Beta sentence one."]],
    ]
    supporting_facts = supporting_facts or [["Alpha", 1], ["Beta", 0]]
    evidences = evidences or [["Alpha", "related_to", "Beta"]]
    return {
        "_id": identity,
        "type": question_type,
        "question": f"Question {identity}?",
        "answer": answer,
        "context": json.dumps(context),
        "supporting_facts": json.dumps(supporting_facts),
        "evidences": json.dumps(evidences),
    }


def _write_sample_manifest(path: Path, selected_ids: list[str]) -> dict:
    payload = {
        "dataset": "2wiki",
        "split": "validation",
        "requested_size": len(selected_ids),
        "selected_ids": selected_ids,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {**payload, "digest": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tracked_sample_manifest_has_100_unique_valid_ids() -> None:
    manifest = two_wiki.load_sample_manifest(SAMPLE_MANIFEST)

    assert manifest["requested_size"] == 100
    assert len(manifest["selected_ids"]) == 100
    assert len(set(manifest["selected_ids"])) == 100
    assert manifest["digest"] == (
        "aa2863cfec1b733ccd96c1a54a764ad60363489846627803cf76c101d9e15539"
    )


def test_load_sample_manifest_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "sample_manifest.json"
    manifest = _write_sample_manifest(path, ["q1", "q2"])
    manifest["selected_ids"][0] = "changed"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        two_wiki.load_sample_manifest(path)


def test_load_selected_rows_uses_pinned_source_and_manifest_order() -> None:
    calls = []

    def fake_loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [_row("q2"), _row("unused"), _row("q1")]

    rows = two_wiki.load_selected_rows(
        ["q1", "q2"],
        load_dataset_fn=fake_loader,
    )

    assert [row["_id"] for row in rows] == ["q1", "q2"]
    assert calls == [
        (
            ("parquet",),
            {
                "data_files": {"dev": two_wiki.SOURCE_URL},
                "split": "dev",
                "streaming": True,
            },
        )
    ]
    assert two_wiki.REVISION in two_wiki.SOURCE_URL


def test_load_selected_rows_reports_missing_ids() -> None:
    with pytest.raises(ValueError, match="were not found"):
        two_wiki.load_selected_rows(
            ["q1", "missing"],
            load_dataset_fn=lambda *args, **kwargs: [_row("q1")],
        )


def test_build_records_creates_shared_corpus_and_full_supervision() -> None:
    first = _row("q1", answer="yes", question_type="comparison")
    second = _row(
        "q2",
        context=[
            ["Alpha", ["Alpha sentence zero.", "Alpha sentence one."]],
            ["Gamma", ["Gamma sentence zero."]],
        ],
        supporting_facts=[["Gamma", 0]],
        evidences=[["Gamma", "instance_of", "Example"]],
    )

    corpus, tests = two_wiki.build_records([second, first])

    assert {document["title"] for document in corpus} == {"Alpha", "Beta", "Gamma"}
    alpha = next(document for document in corpus if document["title"] == "Alpha")
    assert alpha["source_question_ids"] == ["q1", "q2"]
    assert alpha["text"] == "Alpha sentence zero. Alpha sentence one."
    assert [example["id"] for example in tests] == ["q1", "q2"]
    assert tests[0]["answer_type"] == "yes"
    alpha_id, beta_id = tests[0]["candidate_document_ids"]
    assert tests[0]["relevant_document_ids"] == [alpha_id, beta_id]
    assert tests[0]["supporting_facts"] == [
        {"document_id": alpha_id, "sentence_id": 1},
        {"document_id": beta_id, "sentence_id": 0},
    ]
    assert tests[1]["evidences"] == [
        {"subject": "Gamma", "relation": "instance_of", "object": "Example"}
    ]


def test_build_records_accepts_column_mapping_fields() -> None:
    row = _row("q1")
    row["context"] = {
        "title": ["Alpha", "Beta"],
        "content": [["A0", "A1"], ["B0"]],
    }
    row["supporting_facts"] = {
        "title": ["Alpha", "Beta"],
        "sent_id": [1, 0],
    }
    row["evidences"] = {
        "fact": ["Alpha"],
        "relation": ["related_to"],
        "entity": ["Beta"],
    }

    corpus, tests = two_wiki.build_records([row])

    assert len(corpus) == 2
    assert len(tests[0]["supporting_facts"]) == 2
    assert len(tests[0]["evidences"]) == 1


def test_build_records_preserves_duplicate_context_titles_with_suffixes() -> None:
    row = _row(
        "q1",
        context=[
            ["Alpha", ["A0", "A1"]],
            ["Alpha", ["A0", "A1"]],
        ],
        supporting_facts=[["Alpha", 1]],
    )

    corpus, tests = two_wiki.build_records([row])

    document_ids = tests[0]["candidate_document_ids"]
    assert len(corpus) == 2
    assert document_ids[1] == f"{document_ids[0]}#2"
    assert tests[0]["relevant_document_ids"] == document_ids
    assert tests[0]["supporting_facts"] == [
        {"document_id": document_ids[0], "sentence_id": 1},
        {"document_id": document_ids[1], "sentence_id": 1},
    ]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            _row("bad-json") | {"context": "not-json"},
            "invalid JSON in context",
        ),
        (
            _row("missing-title", supporting_facts=[["Missing", 0]]),
            "support title is absent",
        ),
        (
            _row("bad-sentence", supporting_facts=[["Alpha", 9]]),
            "out of range",
        ),
        (
            _row("empty-answer") | {"answer": "  "},
            "missing non-empty 'answer'",
        ),
    ],
)
def test_build_records_rejects_invalid_source_rows(row, message) -> None:
    with pytest.raises(ValueError, match=message):
        two_wiki.build_records([row])


def test_build_records_preserves_different_snapshots_with_the_same_title() -> None:
    first = _row("q1")
    second = _row(
        "q2",
        context=[
            ["Alpha", ["Different text."]],
            ["Gamma", ["Gamma text."]],
        ],
        supporting_facts=[["Gamma", 0]],
    )

    corpus, tests = two_wiki.build_records([first, second])

    alpha_documents = [document for document in corpus if document["title"] == "Alpha"]
    assert len(alpha_documents) == 2
    assert len({document["id"] for document in alpha_documents}) == 2
    assert len(tests) == 2


def test_build_demo_writes_reproducible_files_and_hashes(tmp_path) -> None:
    sample_manifest_path = tmp_path / "sample_manifest.json"
    _write_sample_manifest(sample_manifest_path, ["q2", "q1"])
    rows = [_row("q1"), _row("q2")]

    def fake_loader(*args, **kwargs):
        return rows

    output = tmp_path / "demo"
    first = two_wiki.build_demo(
        sample_manifest_path,
        output,
        load_dataset_fn=fake_loader,
    )
    second = two_wiki.build_demo(
        sample_manifest_path,
        output,
        load_dataset_fn=fake_loader,
    )

    assert first == second
    assert first["counts"] == {
        "documents": 2,
        "test_examples": 2,
        "supporting_facts": 4,
        "evidences": 2,
    }
    assert first["sampling"]["type_counts"] == {"compositional": 2}
    assert len(_load_jsonl(output / "corpus.jsonl")) == 2
    assert len(_load_jsonl(output / "test.jsonl")) == 2
    for filename in ("corpus.jsonl", "test.jsonl"):
        assert first["files"][filename]["sha256"] == _sha256(output / filename)

    (output / "test.jsonl").write_text("different\n", encoding="utf-8")
    with pytest.raises(two_wiki.DatasetStateError, match="--force"):
        two_wiki.build_demo(
            sample_manifest_path,
            output,
            load_dataset_fn=fake_loader,
        )

    rebuilt = two_wiki.build_demo(
        sample_manifest_path,
        output,
        force=True,
        load_dataset_fn=fake_loader,
    )
    assert rebuilt == first
    assert len(_load_jsonl(output / "test.jsonl")) == 2


def test_local_generated_demo_is_internally_consistent_when_present() -> None:
    required = [
        LOCAL_DEMO_ROOT / "corpus.jsonl",
        LOCAL_DEMO_ROOT / "test.jsonl",
        LOCAL_DEMO_ROOT / "manifest.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Local generated 2Wiki demo files are intentionally not tracked")

    corpus_records = _load_jsonl(required[0])
    corpus = {record["id"]: record for record in corpus_records}
    tests = _load_jsonl(required[1])
    manifest = json.loads(required[2].read_text(encoding="utf-8"))

    assert len(corpus) == len(corpus_records) == manifest["counts"]["documents"]
    assert len(tests) == manifest["counts"]["test_examples"] == 100
    assert Counter(record["type"] for record in tests) == Counter(
        manifest["sampling"]["type_counts"]
    )
    for filename, path in zip(
        ("corpus.jsonl", "test.jsonl"), required[:2], strict=True
    ):
        assert _sha256(path) == manifest["files"][filename]["sha256"]
    for example in tests:
        candidates = set(example["candidate_document_ids"])
        relevant = set(example["relevant_document_ids"])
        assert candidates <= corpus.keys()
        assert relevant <= candidates
        assert relevant
        for fact in example["supporting_facts"]:
            document = corpus[fact["document_id"]]
            assert 0 <= fact["sentence_id"] < len(document["sentences"])
