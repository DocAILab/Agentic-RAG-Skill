"""Build a reproducible FinanceBench evidence-page retrieval demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.retrieval.adapters import adapt_financebench

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "FinanceBench" / "small" / "train.jsonl"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data" / "demo"
DEFAULT_SEED = 20260831
DEFAULT_CANDIDATE_COUNT = 10
REPOSITORY = "PatronusAI/financebench"
REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
LICENSE = "CC BY-NC 4.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"FinanceBench source does not exist: {path}. Run the data loader first."
        )
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"Record at {path}:{line_number} must be an object")
        rows.append(dict(row))
    if not rows:
        raise ValueError("FinanceBench source contains no examples")
    return rows


def select_rows(
    rows: Sequence[Mapping[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if limit is None:
        return [dict(row) for row in rows]
    if limit < 1:
        raise ValueError("limit must be positive")
    if limit > len(rows):
        raise ValueError(f"limit {limit} exceeds the {len(rows)} available examples")
    ranked = sorted(rows, key=lambda row: _selection_key(_row_id(row), seed))
    return [dict(row) for row in ranked[:limit]]


def build_records(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    test_rows: Sequence[Mapping[str, Any]] | None = None,
    seed: int = DEFAULT_SEED,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a shared corpus and deterministic per-question candidate pools."""
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    corpus_by_id: dict[str, dict[str, Any]] = {}
    source_questions: dict[str, set[str]] = defaultdict(set)
    document_companies: dict[str, set[str]] = defaultdict(set)
    adapted_by_id = {}

    for row in source_rows:
        example = adapt_financebench(row)
        if example.id in adapted_by_id:
            raise ValueError(f"Duplicate FinanceBench question ID: {example.id}")
        adapted_by_id[example.id] = example
        company = str(row.get("company") or "").strip()
        for document in example.documents:
            record = {
                "id": document.id,
                "title": document.title,
                "text": document.text,
            }
            existing = corpus_by_id.get(document.id)
            if existing is not None and existing != record:
                raise ValueError(
                    f"Conflicting content for FinanceBench page: {document.id}"
                )
            corpus_by_id[document.id] = record
            source_questions[document.id].add(example.id)
            if company:
                document_companies[document.id].add(company)

    if len(corpus_by_id) < candidate_count:
        raise ValueError(
            f"Cannot build {candidate_count} candidates from "
            f"{len(corpus_by_id)} unique pages"
        )
    selected = source_rows if test_rows is None else test_rows
    tests = []
    seen_test_ids = set()
    for row in selected:
        identity = _row_id(row)
        example = adapted_by_id.get(identity)
        if example is None:
            raise ValueError(
                f"Selected question is absent from source rows: {identity}"
            )
        if identity in seen_test_ids:
            raise ValueError(f"Duplicate selected question ID: {identity}")
        seen_test_ids.add(identity)
        relevant = list(example.relevant_document_ids)
        candidate_ids = _candidate_ids(
            identity,
            relevant,
            corpus_by_id,
            document_companies,
            company=str(row.get("company") or "").strip(),
            seed=seed,
            candidate_count=candidate_count,
        )
        answer = str(row.get("answer") or "").strip()
        if not answer:
            raise ValueError(f"Question {identity} has no answer")
        tests.append(
            {
                "id": identity,
                "question": example.query,
                "answer": answer,
                "answers": [answer],
                "candidate_document_ids": candidate_ids,
                "relevant_document_ids": relevant,
                "company": row.get("company"),
                "doc_name": row.get("doc_name"),
                "question_type": row.get("question_type"),
            }
        )

    corpus = []
    for document_id in sorted(corpus_by_id):
        record = dict(corpus_by_id[document_id])
        record["companies"] = sorted(document_companies[document_id])
        record["source_question_ids"] = sorted(source_questions[document_id])
        corpus.append(record)
    return corpus, sorted(tests, key=lambda record: record["id"])


def write_demo(
    *,
    source: Path,
    source_manifest: Mapping[str, Any],
    output: Path,
    seed: int,
    limit: int | None,
    corpus: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "corpus.jsonl"
    test_path = output / "test.jsonl"
    _write_jsonl(corpus_path, corpus)
    _write_jsonl(test_path, tests)
    source_manifest_path = source.parent / "manifest.json"
    manifest = {
        "schema_version": 1,
        "name": "financebench-evidence-page-demo-v2",
        "source": {
            "dataset": "FinanceBench",
            "repository": REPOSITORY,
            "revision": REVISION,
            "split": "train",
            "path": _repo_relative(source),
            "sha256": _sha256(source),
            "manifest_path": _repo_relative(source_manifest_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "dataset_version": source_manifest["name"],
            "license": LICENSE,
        },
        "corpus": {
            "scope": "closed-evidence-pages",
            "description": (
                "Unique annotated full evidence pages; this is not the complete "
                "PDF corpus."
            ),
        },
        "candidates": {
            "size": DEFAULT_CANDIDATE_COUNT,
            "method": "tiered-sha256-hard-negatives",
            "negative_priority": ["same-document", "same-company", "other"],
            "ordering": "sha256-question-document",
        },
        "sampling": {
            "method": "all" if limit is None else "sha256-id-ranking",
            "seed": seed,
            "limit": limit,
        },
        "counts": {
            "documents": len(corpus),
            "test_examples": len(tests),
            "relevant_pages": sum(
                len(row["relevant_document_ids"]) for row in tests
            ),
        },
        "files": {
            "corpus.jsonl": _output_record(corpus_path, len(corpus)),
            "test.jsonl": _output_record(test_path, len(tests)),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_demo(
    source: Path,
    output: Path,
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source = Path(source)
    source_manifest = _read_source_manifest(source)
    rows = load_rows(source)
    selected = select_rows(rows, limit=limit, seed=seed)
    corpus, tests = build_records(rows, test_rows=selected, seed=seed)
    return write_demo(
        source=source,
        source_manifest=source_manifest,
        output=Path(output),
        seed=seed,
        limit=limit,
        corpus=corpus,
        tests=tests,
    )


def _read_source_manifest(source: Path) -> dict[str, Any]:
    path = source.parent / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"FinanceBench source manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_info = manifest.get("source", {})
    if source_info.get("repository") != REPOSITORY:
        raise ValueError("FinanceBench source manifest has an unexpected repository")
    if source_info.get("revision") != REVISION:
        raise ValueError("FinanceBench source manifest has an unexpected revision")
    records = manifest.get("files", [])
    source_record = next(
        (record for record in records if record.get("path") == source.name), None
    )
    if source_record is None or source_record.get("sha256") != _sha256(source):
        raise ValueError("FinanceBench source does not match its manifest")
    return manifest


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("financebench_id") or row.get("id") or row.get("question_id")
    if value is None or not str(value).strip():
        raise ValueError("FinanceBench row has no stable question ID")
    return str(value).strip()


def _selection_key(identity: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _candidate_ids(
    question_id: str,
    relevant_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
    document_companies: Mapping[str, set[str]],
    *,
    company: str,
    seed: int,
    candidate_count: int,
) -> list[str]:
    relevant = list(dict.fromkeys(relevant_ids))
    if not relevant or not set(relevant) <= set(corpus):
        raise ValueError(f"Question {question_id} has invalid relevant pages")
    if len(relevant) >= candidate_count:
        raise ValueError(
            f"Question {question_id} has {len(relevant)} relevant pages, "
            f"leaving no room in a {candidate_count}-page candidate pool"
        )
    relevant_titles = {str(corpus[item]["title"]) for item in relevant}

    def negative_key(document_id: str) -> tuple[int, str, str]:
        document = corpus[document_id]
        if str(document["title"]) in relevant_titles:
            tier = 0
        elif company and company in document_companies.get(document_id, set()):
            tier = 1
        else:
            tier = 2
        digest = hashlib.sha256(
            f"negative:{seed}:{question_id}:{document_id}".encode()
        ).hexdigest()
        return tier, digest, document_id

    relevant_set = set(relevant)
    negatives = sorted(
        (document_id for document_id in corpus if document_id not in relevant_set),
        key=negative_key,
    )[: candidate_count - len(relevant)]
    selected = [*relevant, *negatives]
    return sorted(
        selected,
        key=lambda document_id: (
            hashlib.sha256(
                f"order:{seed}:{question_id}:{document_id}".encode()
            ).hexdigest(),
            document_id,
        ),
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def _output_record(path: Path, count: int) -> dict[str, Any]:
    return {"records": count, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_demo(
        args.source,
        args.output,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
