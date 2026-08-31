"""Build a reproducible 2WikiMultihopQA demo corpus and test set."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DATA_ROOT.parents[1]
DEFAULT_SAMPLE_MANIFEST = DATA_ROOT / "demo" / "sample_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "demo"

REPOSITORY = "xanhho/2WikiMultihopQA"
REVISION = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
SOURCE_SPLIT = "dev"
SOURCE_FILE = "dev.parquet"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
    f"{REVISION}/{SOURCE_FILE}"
)
DEMO_NAME = "2wikimultihopqa-demo-v1"
GENERATED_FILES = ("corpus.jsonl", "test.jsonl", "manifest.json")


class DatasetStateError(RuntimeError):
    """Raised when existing demo outputs disagree with a requested build."""


def load_sample_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a manifest produced by retrieval.run_manifest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"2Wiki sample manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("digest", None)
    expected_digest = _manifest_digest(payload)
    if digest != expected_digest:
        raise ValueError("2Wiki sample manifest digest does not match its contents")
    if payload.get("dataset") not in {"2wiki", "2wikimultihopqa"}:
        raise ValueError("sample manifest dataset must be 2wiki")
    if payload.get("split") not in {"validation", "dev"}:
        raise ValueError("sample manifest split must be validation or dev")
    selected_ids = payload.get("selected_ids")
    requested_size = payload.get("requested_size")
    if not _is_sequence(selected_ids) or not selected_ids:
        raise ValueError("sample manifest must contain selected_ids")
    normalized_ids = [str(value).strip() for value in selected_ids]
    if any(not value for value in normalized_ids):
        raise ValueError("sample manifest contains an empty selected id")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("sample manifest contains duplicate selected ids")
    if requested_size != len(normalized_ids):
        raise ValueError("sample manifest requested_size does not match selected_ids")
    return {
        **payload,
        "selected_ids": normalized_ids,
        "digest": digest,
    }


def load_selected_rows(
    selected_ids: Sequence[str],
    *,
    load_dataset_fn=None,
) -> list[dict[str, Any]]:
    """Stream the pinned dev shard and return selected rows in manifest order."""
    selected = list(selected_ids)
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("selected 2Wiki ids must be unique")
    loader = load_dataset_fn or _load_dataset
    rows = loader(
        "parquet",
        data_files={SOURCE_SPLIT: SOURCE_URL},
        split=SOURCE_SPLIT,
        streaming=True,
    )
    found: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = _required_text(row, "_id", "<unknown>")
        if identity not in selected_set:
            continue
        if identity in found:
            raise ValueError(f"duplicate 2Wiki sample id in source: {identity}")
        found[identity] = dict(row)
        if len(found) == len(selected):
            break
    missing = selected_set - found.keys()
    if missing:
        raise ValueError(f"2Wiki manifest samples were not found: {sorted(missing)[:3]}")
    return [found[identity] for identity in selected]


def build_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert selected source rows into a shared corpus and RAG test records."""
    corpus_by_key: dict[str, dict[str, Any]] = {}
    source_questions: dict[str, set[str]] = defaultdict(set)
    tests: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for row in rows:
        question_id = _required_text(row, "_id", "<unknown>")
        if question_id in seen_questions:
            raise ValueError(f"duplicate 2Wiki question id: {question_id}")
        seen_questions.add(question_id)
        question = _required_text(row, "question", question_id)
        answer = _required_text(row, "answer", question_id)
        question_type = _required_text(row, "type", question_id)

        context_items = _records(
            row.get("context"),
            ("title", "content"),
            "context",
            question_id,
        )
        candidate_ids: list[str] = []
        local_title_ids: dict[str, list[str]] = defaultdict(list)
        local_title_counts: dict[str, int] = {}
        for index, item in enumerate(context_items):
            title = str(item.get("title") or "").strip()
            if not title:
                raise ValueError(
                    f"2Wiki question {question_id} context item {index} has no title"
                )
            sentences = _sentences(item.get("content"), question_id, title)
            normalized_title = _normalize_title(title)
            occurrence = local_title_counts.get(normalized_title, 0) + 1
            local_title_counts[normalized_title] = occurrence
            document_id = _document_id(title, sentences, occurrence=occurrence)
            document_key = _normalize_title(document_id)
            existing = corpus_by_key.get(document_key)
            if existing is not None:
                if existing["sentences"] != sentences:
                    raise AssertionError(
                        f"2Wiki content hash collision for document: {document_id}"
                    )
                document_id = str(existing["id"])
            else:
                corpus_by_key[document_key] = {
                    "id": document_id,
                    "title": title,
                    "text": " ".join(sentence for sentence in sentences if sentence),
                    "sentences": sentences,
                }
            local_title_ids[normalized_title].append(document_id)
            candidate_ids.append(document_id)
            source_questions[document_key].add(question_id)
        if not candidate_ids:
            raise ValueError(f"2Wiki question {question_id} has an empty context")

        supporting_items = _records(
            row.get("supporting_facts"),
            ("title", "sent_id"),
            "supporting_facts",
            question_id,
        )
        supporting_facts: list[dict[str, Any]] = []
        for item in supporting_items:
            title = str(item.get("title") or "").strip()
            normalized_title = _normalize_title(title)
            document_ids = local_title_ids.get(normalized_title)
            if not document_ids:
                raise ValueError(
                    f"2Wiki question {question_id} support title is absent from "
                    f"context: {title}"
                )
            sentence_id = _sentence_id(item.get("sent_id"), question_id, title)
            for document_id in document_ids:
                document = corpus_by_key[_normalize_title(document_id)]
                if sentence_id >= len(document["sentences"]):
                    raise ValueError(
                        f"2Wiki question {question_id} support sentence is out of "
                        f"range: {document_id}[{sentence_id}]"
                    )
                supporting_facts.append(
                    {"document_id": document_id, "sentence_id": sentence_id}
                )
        if not supporting_facts:
            raise ValueError(f"2Wiki question {question_id} has no supporting facts")

        evidences = _build_evidences(row.get("evidences"), question_id)
        relevant_ids = list(
            dict.fromkeys(fact["document_id"] for fact in supporting_facts)
        )
        tests.append(
            {
                "id": question_id,
                "question": question,
                "answer": answer,
                "answers": [answer],
                "type": question_type,
                "answer_type": _answer_type(answer),
                "relevant_document_ids": relevant_ids,
                "supporting_facts": supporting_facts,
                "candidate_document_ids": candidate_ids,
                "evidences": evidences,
            }
        )

    corpus: list[dict[str, Any]] = []
    for normalized_title in sorted(corpus_by_key):
        document = dict(corpus_by_key[normalized_title])
        document["source_question_ids"] = sorted(source_questions[normalized_title])
        corpus.append(document)
    return corpus, sorted(tests, key=lambda record: record["id"])


def write_demo(
    output: Path,
    *,
    sample_manifest_path: Path,
    sample_manifest: Mapping[str, Any],
    corpus: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    force: bool = False,
) -> dict[str, Any]:
    """Write stable JSONL files and an integrity manifest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "corpus.jsonl"
    test_path = output / "test.jsonl"
    _write_stable(
        corpus_path,
        _jsonl_content(corpus),
        force=force,
    )
    _write_stable(
        test_path,
        _jsonl_content(tests),
        force=force,
    )

    type_counts = Counter(str(record["type"]) for record in tests)
    answer_type_counts = Counter(str(record["answer_type"]) for record in tests)
    manifest = {
        "schema_version": 1,
        "name": DEMO_NAME,
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "split": SOURCE_SPLIT,
            "file": SOURCE_FILE,
            "url": SOURCE_URL,
            "license": "Apache-2.0",
        },
        "sampling": {
            "method": "stable-sha256-id-manifest",
            "manifest": _repo_relative(sample_manifest_path),
            "manifest_digest": sample_manifest["digest"],
            "requested_size": sample_manifest["requested_size"],
            "type_counts": dict(sorted(type_counts.items())),
            "answer_type_counts": dict(sorted(answer_type_counts.items())),
        },
        "counts": {
            "documents": len(corpus),
            "test_examples": len(tests),
            "supporting_facts": sum(
                len(record["supporting_facts"]) for record in tests
            ),
            "evidences": sum(len(record["evidences"]) for record in tests),
        },
        "files": {
            "corpus.jsonl": _file_record(corpus_path, len(corpus)),
            "test.jsonl": _file_record(test_path, len(tests)),
        },
    }
    manifest_path = output / "manifest.json"
    _write_stable(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        force=force,
    )
    return manifest


def build_demo(
    sample_manifest_path: Path = DEFAULT_SAMPLE_MANIFEST,
    output: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
    load_dataset_fn=None,
) -> dict[str, Any]:
    """Run manifest validation, source loading, conversion, and stable writes."""
    sample_manifest = load_sample_manifest(sample_manifest_path)
    rows = load_selected_rows(
        sample_manifest["selected_ids"],
        load_dataset_fn=load_dataset_fn,
    )
    corpus, tests = build_records(rows)
    return write_demo(
        output,
        sample_manifest_path=sample_manifest_path,
        sample_manifest=sample_manifest,
        corpus=corpus,
        tests=tests,
        force=force,
    )


def _records(
    value: Any,
    fields: tuple[str, ...],
    label: str,
    question_id: str,
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"2Wiki question {question_id} has invalid JSON in {label}"
            ) from exc
    if isinstance(value, Mapping):
        columns = {field: value.get(field) for field in fields}
        if any(not _is_sequence(column) for column in columns.values()):
            raise ValueError(
                f"2Wiki question {question_id} has invalid {label} columns"
            )
        lengths = {len(column) for column in columns.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"2Wiki question {question_id} has inconsistent {label} columns"
            )
        return [
            {field: columns[field][index] for field in fields}
            for index in range(next(iter(lengths), 0))
        ]
    if not _is_sequence(value):
        raise ValueError(f"2Wiki question {question_id} has invalid {label}")
    records = []
    for item in value:
        if isinstance(item, Mapping):
            records.append(dict(item))
        elif _is_sequence(item):
            if len(item) != len(fields):
                raise ValueError(
                    f"2Wiki question {question_id} has malformed {label} record"
                )
            records.append(dict(zip(fields, item, strict=True)))
        else:
            raise ValueError(
                f"2Wiki question {question_id} has malformed {label} record"
            )
    return records


def _build_evidences(value: Any, question_id: str) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"2Wiki question {question_id} has invalid JSON in evidences"
            ) from exc
    fields = (
        ("fact", "relation", "entity")
        if isinstance(value, Mapping) and ("fact" in value or "entity" in value)
        else ("subject", "relation", "object")
    )
    items = _records(
        value,
        fields,
        "evidences",
        question_id,
    )
    evidences = []
    for item in items:
        evidence = {
            "subject": str(item.get("subject") or item.get("fact") or "").strip(),
            "relation": str(item.get("relation") or "").strip(),
            "object": str(item.get("object") or item.get("entity") or "").strip(),
        }
        if any(not value for value in evidence.values()):
            raise ValueError(
                f"2Wiki question {question_id} has an empty evidence field"
            )
        evidences.append(evidence)
    return evidences


def _sentences(value: Any, question_id: str, title: str) -> list[str]:
    if not _is_sequence(value):
        raise ValueError(
            f"2Wiki question {question_id} has invalid sentences for {title}"
        )
    sentences = [str(sentence).strip() for sentence in value]
    if not sentences or not any(sentences):
        raise ValueError(
            f"2Wiki question {question_id} has empty sentences for {title}"
        )
    return sentences


def _sentence_id(value: Any, question_id: str, title: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"2Wiki question {question_id} has invalid sentence id for {title}"
        )
    try:
        sentence_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"2Wiki question {question_id} has invalid sentence id for {title}"
        ) from exc
    if sentence_id < 0:
        raise ValueError(
            f"2Wiki question {question_id} has negative sentence id for {title}"
        )
    return sentence_id


def _required_text(row: Mapping[str, Any], key: str, identity: str) -> str:
    value = row.get(key)
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"2Wiki question {identity} is missing non-empty '{key}'")
    return text


def _answer_type(answer: str) -> str:
    normalized = answer.casefold().strip()
    return normalized if normalized in {"yes", "no"} else "span"


def _normalize_title(title: str) -> str:
    return unicodedata.normalize("NFKC", title).casefold().strip()


def _document_id(title: str, sentences: Sequence[str], *, occurrence: int) -> str:
    encoded = json.dumps(
        list(sentences),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    base = f"{title}@{digest}"
    return base if occurrence == 1 else f"{base}#{occurrence}"


def _jsonl_content(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _write_stable(path: Path, content: str, *, force: bool) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        if not force:
            raise DatasetStateError(
                f"refusing to overwrite different 2Wiki output: {path}; "
                "rerun with --force"
            )
    path.write_text(content, encoding="utf-8")


def _file_record(path: Path, records: int) -> dict[str, Any]:
    return {
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "2Wiki demo preparation requires the project experiment dependencies"
        ) from exc
    return load_dataset(*args, **kwargs)
