"""Load and validate immutable scene corpora and their QA tasks.

Preparation annotations remain in files. Runtime documents expose only their
opaque identity, title and text; answers and relevance remain on QA objects for
the evaluator and are never attached to the corpus passed to Skills.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Any, Iterator


class GroupDataError(ValueError):
    """A Group manifest or its data violates the runtime data contract."""


@dataclass(frozen=True, slots=True)
class GroupQA:
    id: str
    question: str
    answers: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    label_type: str


@dataclass(frozen=True, slots=True)
class GroupTask:
    task_id: str
    path: Path
    examples: tuple[GroupQA, ...]


@dataclass(frozen=True, slots=True)
class DataGroup:
    group_id: str
    dataset: str
    manifest_path: Path
    corpus_fingerprint: str
    documents: tuple[Mapping[str, str], ...]
    tasks: tuple[GroupTask, ...]
    shuffle_seed: int

    @property
    def qa_count(self) -> int:
        return sum(len(task.examples) for task in self.tasks)


def load_group(path: str | Path, *, shuffle_seed: int = 42) -> DataGroup:
    """Validate all files once and return one shuffled, shared runtime corpus."""
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
        raise GroupDataError("shuffle_seed must be an integer")
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "group.json"
    if not manifest_path.is_file():
        raise GroupDataError(f"Group manifest does not exist: {manifest_path}")
    manifest = _read_object(manifest_path)
    version = manifest.get("schema_version")
    if type(version) is not int or version != 1:
        raise GroupDataError("Only Group schema_version 1 is supported")
    group_id = _text(manifest, "group_id", str(manifest_path))
    dataset = _text(manifest, "dataset", str(manifest_path))
    corpus = _mapping(manifest.get("corpus"), "corpus")
    if corpus.get("unit") != "document":
        raise GroupDataError("corpus.unit must be 'document'")
    root = manifest_path.parent
    file_paths: set[Path] = set()
    documents: dict[str, Mapping[str, str]] = {}
    hashes: set[str] = set()
    role_references: set[str] = set()
    for entry in _entries(corpus.get("files"), "corpus.files"):
        data_path = _checked_file(root, entry, file_paths)
        actual_count = 0
        for record, location in _records(data_path):
            actual_count += 1
            document, digest = _document(record, location)
            identity = document["id"]
            if identity in documents:
                raise GroupDataError(f"Duplicate corpus document ID at {location}: {identity}")
            documents[identity] = MappingProxyType(document)
            hashes.add(digest)
            if record["role"] == "reference":
                role_references.add(identity)
        _equal_count(entry, "records", actual_count, str(data_path))
    document_count = len(documents)
    _equal_count(corpus, "document_count", document_count, "corpus", positive=True)
    _equal_count(corpus, "target_size", document_count, "corpus", positive=True)
    _equal_count(corpus, "reference_count", len(role_references), "corpus")
    _equal_count(corpus, "distractor_count", document_count - len(role_references), "corpus")
    fingerprint = hashlib.sha256(("\n".join(sorted(hashes)) + "\n").encode("utf-8")).hexdigest()
    if _digest(corpus.get("fingerprint"), "corpus.fingerprint") != fingerprint:
        raise GroupDataError("Corpus content fingerprint does not match its documents")

    tasks = []
    task_ids: set[str] = set()
    qa_ids: set[str] = set()
    qa_references: set[str] = set()
    qa_per_task = _integer(manifest.get("qa_per_task"), "qa_per_task", minimum=1)
    for entry in _entries(manifest.get("tasks"), "tasks"):
        task_id = _text(entry, "task_id", "task")
        if task_id in task_ids:
            raise GroupDataError(f"Duplicate task ID: {task_id}")
        task_ids.add(task_id)
        data_path = _checked_file(root, entry, file_paths)
        examples = []
        for record, location in _records(data_path):
            if "documents" in record:
                raise GroupDataError(f"Group QA must reference its shared corpus, not embed documents: {location}")
            identity = _text(record, "id", location)
            if identity in qa_ids:
                raise GroupDataError(f"Duplicate QA ID within Group at {location}: {identity}")
            qa_ids.add(identity)
            relevant = _texts(record.get("relevant_document_ids"), f"{location}:relevant_document_ids")
            missing = set(relevant) - documents.keys()
            if missing:
                raise GroupDataError(f"QA references missing corpus documents at {location}: {sorted(missing)[:3]}")
            qa_references.update(relevant)
            label_type = _text(record, "label_type", location)
            if label_type not in {"supporting_facts", "weak_answer_alias", "evidence_page"}:
                raise GroupDataError(f"Unsupported relevance label type at {location}: {label_type}")
            examples.append(GroupQA(
                id=identity,
                question=_text(record, "question", location),
                answers=_texts(record.get("answers"), f"{location}:answers"),
                relevant_document_ids=relevant,
                label_type=label_type,
            ))
        _equal_count(entry, "qa_count", len(examples), task_id, positive=True)
        if len(examples) != qa_per_task:
            raise GroupDataError(f"Task {task_id} has {len(examples)} QA, expected qa_per_task={qa_per_task}")
        tasks.append(GroupTask(task_id, data_path, tuple(examples)))
    _equal_count(manifest, "task_count", len(tasks), "group", positive=True)
    _equal_count(manifest, "qa_count", len(qa_ids), "group", positive=True)
    if qa_references != role_references:
        raise GroupDataError("Corpus reference roles must equal the union of all QA relevance labels")

    shuffled = [documents[identity] for identity in sorted(documents)]
    random.Random(shuffle_seed).shuffle(shuffled)
    return DataGroup(group_id, dataset, manifest_path, fingerprint, tuple(shuffled), tuple(tasks), shuffle_seed)


def _document(record: Mapping[str, Any], location: str) -> tuple[dict[str, str], str]:
    title = record.get("title")
    text = record.get("text")
    if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
        raise GroupDataError(f"Corpus title/text must be strings with nonempty text: {location}")
    title = title.strip()
    text = " ".join(text.split())
    digest = hashlib.sha256(json.dumps([title, text], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if _digest(record.get("content_hash"), f"{location}:content_hash") != digest:
        raise GroupDataError(f"Document content hash mismatch: {location}")
    identity = _text(record, "id", location)
    if identity != "doc-" + digest:
        raise GroupDataError(f"Document ID must use its opaque content hash: {location}")
    if record.get("role") not in {"reference", "distractor"}:
        raise GroupDataError(f"Invalid document role: {location}")
    if record.get("origin") not in {"qa_context", "supplement"}:
        raise GroupDataError(f"Invalid document origin: {location}")
    return {"id": identity, "title": title, "text": text}, digest


def _checked_file(root: Path, entry: Mapping[str, Any], used: set[Path]) -> Path:
    relative = Path(_text(entry, "path", "manifest file entry"))
    if relative.is_absolute() or relative.drive:
        raise GroupDataError(f"Manifest file paths must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise GroupDataError(f"Manifest file path escapes the Group directory: {relative}")
    if path in used:
        raise GroupDataError(f"Manifest lists the same data file twice: {relative}")
    used.add(path)
    if not path.is_file():
        raise GroupDataError(f"Group data file does not exist: {path}")
    if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
        raise GroupDataError(f"Group data must be JSONL or gzipped JSONL: {path}")
    expected = _digest(entry.get("sha256"), f"{relative}:sha256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise GroupDataError(f"Group data file SHA-256 mismatch: {path}")
    return path


def _records(path: Path) -> Iterator[tuple[Mapping[str, Any], str]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GroupDataError(f"Invalid JSON at {location}: {exc}") from exc
                yield _mapping(record, location), location
    except (OSError, EOFError, UnicodeError) as exc:
        raise GroupDataError(f"Cannot read Group data file {path}: {exc}") from exc


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GroupDataError(f"Cannot read Group manifest {path}: {exc}") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroupDataError(f"{name} must be an object")
    return value


def _entries(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise GroupDataError(f"{name} must be a nonempty list")
    return [_mapping(entry, name) for entry in value]


def _text(record: Mapping[str, Any], key: str, name: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GroupDataError(f"{name}:{key} must be a nonempty string")
    return value.strip()


def _texts(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise GroupDataError(f"{name} must be a nonempty string list")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise GroupDataError(f"{name} must not contain duplicates")
    return normalized


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GroupDataError(f"{name} must be an integer >= {minimum}")
    return value


def _equal_count(record: Mapping[str, Any], key: str, actual: int, name: str, *, positive: bool = False) -> None:
    expected = _integer(record.get(key), f"{name}:{key}", minimum=int(positive))
    if expected != actual:
        raise GroupDataError(f"{name}:{key} declares {expected}, found {actual}")


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GroupDataError(f"{name} must be a lowercase SHA-256 hex digest")
    return value
