"""Hugging Face 流式数据加载与逐样本错误隔离。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import (
    AdapterError,
    adapt_financebench,
    adapt_hotpotqa,
    adapt_triviaqa,
    adapt_two_wiki,
)
from .adapters.common import sample_id
from .schema import RetrievalDocument, RetrievalExample

Adapter = Callable[[Mapping[str, Any]], RetrievalExample]

DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", adapt_hotpotqa),
    "2wikimultihopqa": ("xanhho/2WikiMultihopQA", None, adapt_two_wiki),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc", adapt_triviaqa),
    "financebench": ("PatronusAI/financebench", None, adapt_financebench),
}
TWO_WIKI_FILES = {
    split: (
        "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/"
        f"resolve/main/{split}.parquet"
    )
    for split in ("train", "dev", "test")
}
FINANCEBENCH_REVISION = (
    "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
)
DEFAULT_FINANCEBENCH_DEMO = (
    Path(__file__).resolve().parents[1] / "financebench" / "data" / "demo"
)


@dataclass(frozen=True, slots=True)
class DatasetItem:
    source_index: int
    sample_id: str
    example: RetrievalExample | None = None
    error: str | None = None


def iter_huggingface_items(
    dataset: str,
    split: str,
    *,
    config: str | None = None,
    load_dataset_fn=None,
) -> Iterator[DatasetItem]:
    """流式加载指定 split，并把坏样本转换为可记录的 DatasetItem。"""
    dataset_key = _dataset_key(dataset)
    path, default_config, adapter = DATASETS[dataset_key]
    loader = load_dataset_fn or _load_dataset
    if dataset_key == "2wikimultihopqa" and split == "validation":
        resolved_split = "dev"
    elif dataset_key == "financebench" and split != "train":
        raise ValueError("FinanceBench Hugging Face source only provides split 'train'")
    else:
        resolved_split = split
    rows = _load_rows(
        loader,
        dataset_key=dataset_key,
        path=path,
        config=config if config is not None else default_config,
        split=resolved_split,
        revision=(FINANCEBENCH_REVISION if dataset_key == "financebench" else None),
    )
    for index, row in enumerate(rows):
        identity = sample_id(row)
        try:
            yield DatasetItem(index, identity, example=adapter(row))
        except AdapterError as exc:
            yield DatasetItem(index, exc.sample_id, error=str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            yield DatasetItem(index, identity, error=f"sample {identity}: {exc}")


def iter_dataset_items(
    dataset: str,
    split: str,
    *,
    config: str | None = None,
    data_dir: str | Path | None = None,
    load_dataset_fn=None,
) -> Iterator[DatasetItem]:
    """Load the experiment contract appropriate for a dataset."""
    if _dataset_key(dataset) == "financebench":
        if split != "test":
            raise ValueError("FinanceBench retrieval demo only provides split 'test'")
        yield from iter_demo_items(data_dir or DEFAULT_FINANCEBENCH_DEMO)
        return
    yield from iter_huggingface_items(
        dataset,
        split,
        config=config,
        load_dataset_fn=load_dataset_fn,
    )


def iter_demo_items(data_dir: str | Path) -> Iterator[DatasetItem]:
    """Load a corpus/test JSONL demo after validating its manifest hashes."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("name") != "financebench-evidence-page-demo-v2":
        raise ValueError(f"Unexpected FinanceBench demo manifest: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("FinanceBench demo manifest has no files mapping")
    corpus_path = root / "corpus.jsonl"
    test_path = root / "test.jsonl"
    _validate_demo_file(corpus_path, files.get("corpus.jsonl"))
    _validate_demo_file(test_path, files.get("test.jsonl"))
    corpus = _read_corpus(corpus_path)
    candidate_size = manifest.get("candidates", {}).get("size")
    if not isinstance(candidate_size, int) or candidate_size < 2:
        raise ValueError("FinanceBench demo manifest has invalid candidate size")

    for index, row in enumerate(_read_jsonl(test_path)):
        identity = sample_id(row)
        try:
            query = str(row.get("question") or "").strip()
            if not query:
                raise ValueError("missing non-empty 'question'")
            candidate_ids = _string_ids(row, "candidate_document_ids")
            relevant_ids = tuple(_string_ids(row, "relevant_document_ids"))
            if len(candidate_ids) != candidate_size:
                raise ValueError(
                    f"expected {candidate_size} candidate pages, "
                    f"found {len(candidate_ids)}"
                )
            missing = set(candidate_ids) - set(corpus)
            if missing:
                raise ValueError(
                    f"candidate pages are absent from corpus: {sorted(missing)[:3]}"
                )
            if not set(relevant_ids) <= set(candidate_ids):
                raise ValueError("relevant pages are not a subset of candidates")
            documents = tuple(corpus[document_id] for document_id in candidate_ids)
            yield DatasetItem(
                index,
                identity,
                example=RetrievalExample(
                    id=identity,
                    query=query,
                    documents=documents,
                    relevant_document_ids=relevant_ids,
                    label_type="evidence_page",
                    metadata={
                        "dataset": "financebench",
                        "corpus_scope": manifest.get("corpus", {}).get("scope"),
                    },
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            yield DatasetItem(index, identity, error=f"sample {identity}: {exc}")


def _dataset_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "")
    aliases = {"2wiki": "2wikimultihopqa", "2wikimultihopqa": "2wikimultihopqa"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DATASETS:
        raise ValueError(f"Unsupported dataset: {value}")
    return normalized


def _load_rows(loader, *, dataset_key, path, config, split, revision):
    if dataset_key == "2wikimultihopqa":
        if split not in TWO_WIKI_FILES:
            raise ValueError(f"Unsupported 2Wiki split: {split}")
        return loader(
            "parquet",
            data_files={split: TWO_WIKI_FILES[split]},
            split=split,
            streaming=True,
        )
    options = {"name": config, "split": split, "streaming": True}
    if revision is not None:
        options["revision"] = revision
    return loader(path, **options)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"FinanceBench demo file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Record at {path}:{line_number} must be an object")
        records.append(dict(payload))
    return records


def _read_corpus(path: Path) -> dict[str, RetrievalDocument]:
    corpus = {}
    for row in _read_jsonl(path):
        identity = str(row.get("id") or "").strip()
        if not identity or identity in corpus:
            raise ValueError(
                f"Corpus has a missing or duplicate document ID: {identity!r}"
            )
        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"Corpus document has no text: {identity}")
        corpus[identity] = RetrievalDocument(
            identity,
            str(row.get("title") or "").strip(),
            text,
        )
    if not corpus:
        raise ValueError("FinanceBench demo corpus is empty")
    return corpus


def _string_ids(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{key}' must be a non-empty list")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"'{key}' contains empty or duplicate IDs")
    return normalized


def _validate_demo_file(path: Path, record: Any) -> None:
    if not isinstance(record, Mapping):
        raise ValueError(f"FinanceBench demo manifest has no record for {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"FinanceBench demo file does not exist: {path}")
    if path.stat().st_size != record.get("bytes"):
        raise ValueError(f"FinanceBench demo file size mismatch: {path}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"FinanceBench demo file checksum mismatch: {path}")
    if len(_read_jsonl(path)) != record.get("records"):
        raise ValueError(f"FinanceBench demo record count mismatch: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Retrieval benchmarks require the 'datasets' package; "
            "install the experiment extra"
        ) from exc
    return load_dataset(*args, **kwargs)
