"""从 HotpotQA distractor validation 构建可复现的小型 demo 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    EXPERIMENT_ROOT
    / "data"
    / "raw"
    / "distractor"
    / "validation-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data" / "demo"
DEFAULT_SEED = 20260807
STRATUM_QUOTAS = {
    ("bridge", "span"): 10,
    ("comparison", "span"): 6,
    ("comparison", "yes"): 2,
    ("comparison", "no"): 2,
}


def parse_args() -> argparse.Namespace:
    """解析源 Parquet、输出目录和确定性采样种子。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_validation_rows(path: Path) -> list[dict[str, Any]]:
    """读取构建 demo 所需的 HotpotQA validation 字段。"""
    if not path.is_file():
        raise FileNotFoundError(f"HotpotQA source file does not exist: {path}")
    columns = [
        "id",
        "question",
        "answer",
        "type",
        "level",
        "supporting_facts",
        "context",
    ]
    return pq.read_table(path, columns=columns).to_pylist()


def select_demo_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """按问题类型和答案类型配额确定性选择 demo 样本。"""
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["type"]), _answer_kind(str(row["answer"])))
        buckets[key].append(row)

    selected: list[dict[str, Any]] = []
    for stratum, quota in STRATUM_QUOTAS.items():
        candidates = buckets.get(stratum, [])
        if len(candidates) < quota:
            raise ValueError(
                f"Stratum {stratum} has {len(candidates)} rows, expected {quota}"
            )
        ranked = sorted(
            candidates,
            key=lambda row: _selection_key(str(row["id"]), seed),
        )
        selected.extend(dict(row) for row in ranked[:quota])
    return sorted(selected, key=lambda row: str(row["id"]))


def build_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造共享小语料库和带检索监督的测试样本。"""
    corpus_by_id: dict[str, dict[str, Any]] = {}
    source_questions: dict[str, set[str]] = defaultdict(set)
    tests: list[dict[str, Any]] = []

    for row in rows:
        question_id = str(row["id"])
        context = _required_mapping(row, "context")
        titles = _required_sequence(context, "title")
        sentence_groups = _required_sequence(context, "sentences")
        if len(titles) != len(sentence_groups):
            raise ValueError(f"Context lengths do not match for question {question_id}")

        candidate_ids: list[str] = []
        for title_value, sentence_values in zip(
            titles,
            sentence_groups,
            strict=True,
        ):
            title = str(title_value)
            sentences = [str(sentence).strip() for sentence in sentence_values]
            document = {
                "id": title,
                "title": title,
                "text": " ".join(sentence for sentence in sentences if sentence),
                "sentences": sentences,
            }
            existing = corpus_by_id.get(title)
            if existing is not None and existing["sentences"] != sentences:
                raise ValueError(f"Conflicting contexts found for title: {title}")
            corpus_by_id[title] = document
            source_questions[title].add(question_id)
            candidate_ids.append(title)

        supporting = _build_supporting_facts(row, question_id)
        relevant_ids = list(
            dict.fromkeys(fact["document_id"] for fact in supporting)
        )
        missing = set(relevant_ids) - set(candidate_ids)
        if missing:
            raise ValueError(
                f"Question {question_id} has missing support documents: {sorted(missing)}"
            )
        answer = str(row["answer"])
        tests.append(
            {
                "id": question_id,
                "question": str(row["question"]),
                "answer": answer,
                "answers": [answer],
                "type": str(row["type"]),
                "level": str(row["level"]),
                "answer_type": _answer_kind(answer),
                "relevant_document_ids": relevant_ids,
                "supporting_facts": supporting,
                "candidate_document_ids": candidate_ids,
            }
        )

    corpus = []
    for document_id in sorted(corpus_by_id):
        document = dict(corpus_by_id[document_id])
        document["source_question_ids"] = sorted(source_questions[document_id])
        corpus.append(document)
    return corpus, sorted(tests, key=lambda record: record["id"])


def write_demo(
    *,
    source: Path,
    output: Path,
    seed: int,
    corpus: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """写入 JSONL 数据和包含校验摘要的 manifest。"""
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "corpus.jsonl"
    tests_path = output / "test.jsonl"
    _write_jsonl(corpus_path, corpus)
    _write_jsonl(tests_path, tests)

    manifest = {
        "schema_version": 1,
        "name": "hotpotqa-demo-v1",
        "source": {
            "dataset": "HotpotQA",
            "configuration": "distractor",
            "split": "validation",
            "path": source.resolve().relative_to(EXPERIMENT_ROOT.resolve()).as_posix(),
            "sha256": _sha256(source),
            "license": "CC BY-SA 4.0",
        },
        "sampling": {
            "method": "deterministic-sha256-stratified",
            "seed": seed,
            "strata": [
                {"type": key[0], "answer_type": key[1], "count": count}
                for key, count in STRATUM_QUOTAS.items()
            ],
        },
        "counts": {
            "documents": len(corpus),
            "test_examples": len(tests),
            "supporting_facts": sum(
                len(record["supporting_facts"]) for record in tests
            ),
        },
        "files": {
            "corpus.jsonl": {
                "records": len(corpus),
                "bytes": corpus_path.stat().st_size,
                "sha256": _sha256(corpus_path),
            },
            "test.jsonl": {
                "records": len(tests),
                "bytes": tests_path.stat().st_size,
                "sha256": _sha256(tests_path),
            },
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_demo(source: Path, output: Path, *, seed: int) -> dict[str, Any]:
    """执行读取、分层采样、语料构建和文件写入的完整流程。"""
    rows = load_validation_rows(source)
    selected = select_demo_rows(rows, seed=seed)
    corpus, tests = build_records(selected)
    return write_demo(
        source=source,
        output=output,
        seed=seed,
        corpus=corpus,
        tests=tests,
    )


def _answer_kind(answer: str) -> str:
    """把标准答案划分为 yes、no 或普通 span。"""
    normalized = answer.strip().lower()
    return normalized if normalized in {"yes", "no"} else "span"


def _selection_key(question_id: str, seed: int) -> str:
    """生成不依赖输入顺序的确定性样本排序键。"""
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def _build_supporting_facts(
    row: Mapping[str, Any],
    question_id: str,
) -> list[dict[str, Any]]:
    """把平行数组形式的 supporting facts 转为显式记录。"""
    payload = _required_mapping(row, "supporting_facts")
    titles = _required_sequence(payload, "title")
    sentence_ids = _required_sequence(payload, "sent_id")
    if len(titles) != len(sentence_ids):
        raise ValueError(
            f"Supporting fact lengths do not match for question {question_id}"
        )
    return [
        {"document_id": str(title), "sentence_id": int(sentence_id)}
        for title, sentence_id in zip(titles, sentence_ids, strict=True)
    ]


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """读取必需的映射字段并提供明确错误。"""
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' must be a mapping")
    return value


def _required_sequence(
    payload: Mapping[str, Any],
    key: str,
) -> Sequence[Any]:
    """读取必需的非字符串序列字段并提供明确错误。"""
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"'{key}' must be a sequence")
    return value


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """以 UTF-8 JSON Lines 格式稳定写入记录。"""
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免一次读取大型源分片。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """执行命令行构建并输出不含样本正文的摘要。"""
    args = parse_args()
    manifest = build_demo(args.source, args.output, seed=args.seed)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
