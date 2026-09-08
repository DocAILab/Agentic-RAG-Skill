"""Materialize reproducible TriviaQA text subsets from a local RC archive.

只负责原始数据读取、预处理、过滤与按规模裁剪，输出标准化文本 JSON 子集，
并写入带记录数/字节数/sha256 校验的 manifest 支持确定性复用；不实现
embedding，不构建向量索引，不执行检索或评测。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _manifest import (
    DatasetStateError,
    existing_manifest,
    file_record,
    sha256_file,
    write_manifest,
)

__all__ = [
    "DatasetStateError",
    "DEFAULT_EVIDENCE",
    "DEFAULT_MAX_QUERY_SAMPLES",
    "DEFAULT_OUTPUT",
    "DEFAULT_SEED",
    "DEFAULT_SOURCE",
    "SCHEMA_VERSION",
    "SUBSET_MANIFEST_NAME",
    "build_sample",
    "build_subsets",
    "collect_valid_samples",
    "load_qa_records",
    "prepare_subsets",
    "read_json",
    "resolve_documents",
    "sha256_file",
]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "triviaqa"
DEFAULT_SOURCE = RAW_ROOT / "qa" / "wikipedia-dev.json"
DEFAULT_EVIDENCE = RAW_ROOT / "evidence"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
DEFAULT_SEED = 20260828
DEFAULT_MAX_QUERY_SAMPLES = (100, 800)
SCHEMA_VERSION = 1
SUBSET_MANIFEST_NAME = "triviaqa-subsets-v1"

FILTER_REASONS = (
    "dirty_record",
    "empty_question_id",
    "empty_query",
    "invalid_answer",
    "empty_answer",
    "no_documents",
    "duplicate_id",
)


def read_json(path: Path) -> Any:
    """读取 JSON 内容，自动兼容普通文本与 gzip 压缩文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"TriviaQA source file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)


def load_qa_records(path: Path) -> list[Any]:
    """读取 TriviaQA 问答 JSON 文件中的 Data 列表。"""
    payload = read_json(path)
    data = payload.get("Data") if isinstance(payload, Mapping) else None
    if not _is_sequence(data):
        raise ValueError(f"TriviaQA file must contain a 'Data' list: {path}")
    return list(data)


def collect_valid_samples(
    records: Sequence[Any],
    evidence_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """过滤脏样本并按 QuestionId 去重，返回有效样本与过滤统计。"""
    stats = {reason: 0 for reason in FILTER_REASONS}
    samples: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            stats["dirty_record"] += 1
            continue
        sample, reason = build_sample(record, evidence_root)
        if sample is None:
            stats[reason] += 1
            continue
        if sample["sample_id"] in samples:
            stats["duplicate_id"] += 1
            continue
        samples[sample["sample_id"]] = sample
    return list(samples.values()), stats


def build_sample(
    article: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """把一条原始问答记录转为标准化样本，失败时返回过滤原因。"""
    question_id = article.get("QuestionId")
    if not isinstance(question_id, str) or not question_id.strip():
        return None, "empty_question_id"
    question = article.get("Question")
    if not isinstance(question, str) or not question.strip():
        return None, "empty_query"
    answer = article.get("Answer")
    if not isinstance(answer, Mapping):
        return None, "invalid_answer"
    value = str(answer.get("Value") or "").strip()
    if not value:
        return None, "empty_answer"
    aliases = _string_list(answer, "Aliases")
    documents = resolve_documents(article, evidence_root)
    if not documents:
        return None, "no_documents"
    return (
        {
            "sample_id": question_id.strip(),
            "query": question.strip(),
            "golden_answer": value,
            "golden_answers": list(dict.fromkeys([value, *aliases])),
            "documents": documents,
        },
        None,
    )


def resolve_documents(
    article: Mapping[str, Any],
    evidence_root: Path,
) -> list[dict[str, Any]]:
    """按题解析 EntityPages 与 SearchResults 引用并读取证据文档。"""
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entries_key, subdir, source in (
        ("EntityPages", "wikipedia", "wikipedia"),
        ("SearchResults", "web", "web"),
    ):
        entries = article.get(entries_key)
        if not _is_sequence(entries):
            continue
        evidence_dir = evidence_root / subdir
        if not evidence_dir.is_dir():
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            filename = entry.get("Filename")
            if not isinstance(filename, str) or not filename:
                continue
            dedupe_key = (source, filename)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            document = parse_evidence_document(
                _evidence_path(evidence_dir, filename),
                source=source,
            )
            if document is None:
                continue
            entry_title = str(entry.get("Title") or "").strip()
            if entry_title:
                document["title"] = entry_title
            documents.append(document)
    return documents


def parse_evidence_document(path: Path, *, source: str) -> dict[str, Any] | None:
    """读取单篇证据文档，返回 {id,title,text,source}，失败时返回 None。"""
    title = ""
    text: str | None = None
    parsed = False
    try:
        payload = read_json(path)
        parsed = True
        if isinstance(payload, Mapping):
            if source == "wikipedia":
                text = _extract_wikipedia_text(payload)
                title = str(payload.get("DocumentTitle") or "").strip()
            else:
                text = _extract_web_text(payload)
                title = str(payload.get("Title") or "").strip()
    except (OSError, UnicodeError, ValueError):
        text = None
    if not text and not parsed:
        text = _read_text_fallback(path)
    if not text:
        return None
    document_id = f"{source}/{_evidence_id(path.name)}"
    return {
        "id": document_id,
        "title": title or _evidence_id(path.name),
        "text": text,
        "source": source,
    }


def _extract_wikipedia_text(payload: Mapping[str, Any]) -> str | None:
    """从维基证据的 Paragraphs/Sentences 抽取完整正文。"""
    paragraphs = payload.get("Paragraphs")
    if not _is_sequence(paragraphs):
        return None
    blocks: list[str] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, Mapping):
            continue
        sentences = paragraph.get("Sentences")
        if not _is_sequence(sentences):
            continue
        sentence_text = " ".join(
            str(sentence).strip() for sentence in sentences if str(sentence).strip()
        )
        if sentence_text:
            blocks.append(sentence_text)
    return "\n\n".join(blocks) if blocks else None


def _extract_web_text(payload: Mapping[str, Any]) -> str | None:
    """从网页证据的 Title/Description 抽取文本。"""
    parts = [
        part.strip()
        for part in (payload.get("Title"), payload.get("Description"))
        if isinstance(part, str) and part.strip()
    ]
    return "\n\n".join(parts) if parts else None


def _read_text_fallback(path: Path) -> str | None:
    """证据文件不是 JSON 时按纯文本读取（兼容 gzip）。"""
    try:
        with path.open("rb") as handle:
            is_gzip = handle.read(2) == b"\x1f\x8b"
    except OSError:
        return None
    try:
        if is_gzip:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                text = handle.read().strip()
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        return text or None
    except (OSError, UnicodeError):
        return None


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    """读取去空白后的非空字符串列表，非法值返回空列表。"""
    values = payload.get(key, ())
    if not _is_sequence(values):
        return []
    return [
        str(value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


def _is_sequence(value: Any) -> bool:
    """判断值是否为非字符串序列。"""
    return not isinstance(value, (str, bytes, bytearray)) and isinstance(value, Sequence)


def _evidence_path(evidence_dir: Path, filename: str) -> Path:
    """返回证据文件路径；Windows 上文件名含非法字符时回退到清洗后的名字。"""
    original = evidence_dir / filename
    if original.is_file():
        return original
    sanitized = evidence_dir / _sanitize_filename(filename)
    return sanitized if sanitized.is_file() else original


def _sanitize_filename(filename: str) -> str:
    """把 Windows 非法字符替换为下划线，保留目录分隔符。"""
    return "/".join(
        re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)
        for part in filename.replace("\\", "/").split("/")
    )


def build_subsets(
    samples: Sequence[Mapping[str, Any]],
    sizes: Sequence[int],
    *,
    seed: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """按确定性排序前缀生成多档子集，规模互不覆盖。"""
    ranked = sorted(
        samples,
        key=lambda sample: _selection_key(str(sample["sample_id"]), seed),
    )
    return [
        (size, [dict(sample) for sample in ranked[:size]])
        for size in sorted(set(sizes))
    ]


def write_subset(
    path: Path,
    subset: Sequence[Mapping[str, Any]],
    *,
    source: Path,
    evidence_root: Path,
    seed: int,
    max_query_samples: int,
    total_valid: int,
    filtered: Mapping[str, int],
) -> dict[str, Any]:
    """写出单档子集 JSON 并返回统计摘要。"""
    documents = [document for sample in subset for document in sample["documents"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "TriviaQA",
        "source": {
            "qa_file": _repo_relative(source),
            "evidence": _repo_relative(evidence_root),
        },
        "sampling": {
            "method": "deterministic-sha256-prefix",
            "seed": seed,
            "max_query_samples": max_query_samples,
            "total_valid_samples": total_valid,
            "filtered": dict(filtered),
        },
        "counts": {
            "samples": len(subset),
            "documents": len(documents),
            "unique_documents": len({document["id"] for document in documents}),
        },
        "samples": [dict(sample) for sample in subset],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload["counts"]


def prepare_subsets(
    source: Path,
    evidence_root: Path,
    output_root: Path,
    sizes: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    load_records_fn=None,
) -> dict:
    """读取、过滤、裁剪并写出请求的子集，带 manifest 校验与复用。"""
    source = Path(source)
    evidence_root = Path(evidence_root)
    output_root = Path(output_root)
    requested = sorted({int(size) for size in sizes})
    if any(size < 1 for size in requested):
        raise ValueError("subset sizes must be positive integers")

    _validate_inputs(source, evidence_root)
    revision = sha256_file(source)
    existing = None
    if not force:
        existing = existing_manifest(output_root, SUBSET_MANIFEST_NAME, revision)
        if existing is not None:
            expected_source = {
                "qa_file": _repo_relative(source),
                "evidence": _repo_relative(evidence_root),
                "seed": seed,
            }
            actual_source = {
                "qa_file": existing.get("source", {}).get("qa_file"),
                "evidence": existing.get("source", {}).get("evidence"),
                "seed": existing.get("sampling", {}).get("seed"),
            }
            if actual_source != expected_source:
                raise DatasetStateError(
                    "Existing subsets use a different qa_file/evidence/seed; "
                    "rerun with --force"
                )
            present = {record["path"] for record in existing["files"]}
            if all(_subset_filename(source, size) in present for size in requested):
                return existing

    loader = load_records_fn or load_qa_records
    records = loader(source)
    samples, filtered = collect_valid_samples(records, evidence_root)
    if not samples:
        raise ValueError(
            "No valid samples remain after filtering. "
            "Note that TriviaQA test splits have no answers and are filtered out."
        )

    subsets = build_subsets(samples, requested, seed=seed)
    valid_records = (
        {record["path"]: record for record in existing["files"]}
        if existing is not None
        else {}
    )
    files: list[dict[str, Any]] = []
    for size, subset in subsets:
        filename = _subset_filename(source, size)
        if not force and filename in valid_records:
            files.append(valid_records[filename])
            continue
        path = output_root / filename
        _ensure_safe_target(path, output_root)
        write_subset(
            path,
            subset,
            source=source,
            evidence_root=evidence_root,
            seed=seed,
            max_query_samples=size,
            total_valid=len(samples),
            filtered=filtered,
        )
        files.append(file_record(path, output_root))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": SUBSET_MANIFEST_NAME,
        "source": {
            "revision": revision,
            "qa_file": _repo_relative(source),
            "qa_file_sha256": revision,
            "evidence": _repo_relative(evidence_root),
        },
        "sampling": {
            "method": "deterministic-sha256-prefix",
            "seed": seed,
        },
        "counts": {
            "samples": len(samples),
            "filtered": dict(filtered),
        },
        "files": files,
    }
    write_manifest(output_root, manifest)
    return manifest


def _validate_inputs(source: Path, evidence_root: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(
            f"TriviaQA QA file does not exist: {source}\n"
            "请先按 README.md 的说明下载并解压原始数据到 data/raw/triviaqa/。"
        )
    if not evidence_root.is_dir():
        raise FileNotFoundError(
            f"TriviaQA evidence directory does not exist: {evidence_root}\n"
            "请确认 evidence/ 下包含 wikipedia/ 与 web/ 两个证据子目录。"
        )
    if not (evidence_root / "wikipedia").is_dir() and not (evidence_root / "web").is_dir():
        raise FileNotFoundError(
            f"Evidence root has no wikipedia/ or web/ subdirectory: {evidence_root}"
        )


def _subset_filename(source: Path, size: int) -> str:
    return f"{_source_slug(source)}_subset_{size}.json"


def _ensure_safe_target(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    root = output_root.resolve()
    if not resolved.is_relative_to(root):
        raise DatasetStateError(f"Refusing to write outside the output root: {path}")


def _selection_key(question_id: str, seed: int) -> str:
    """生成不依赖输入顺序的确定性排序键。"""
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def _source_slug(source: Path) -> str:
    """从问答文件名生成小写输出前缀。"""
    stem = _strip_suffixes(source.name)
    slug = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-")
    return slug or "triviaqa"


def _strip_suffixes(name: str) -> str:
    """去掉 .json.gz 或 .json 后缀。"""
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _evidence_id(filename: str) -> str:
    """从证据文件名生成稳定文档 ID（去掉 .json/.gz 后缀）。"""
    stem = _strip_suffixes(filename)
    return stem or filename


def _repo_relative(path: Path) -> str:
    """返回相对仓库根目录的 posix 路径，仓库外路径原样返回。"""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
