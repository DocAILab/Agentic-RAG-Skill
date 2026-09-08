"""从 TriviaQA 标准化子集构建可复现的小型 demo 数据。

读取 data/TriviaQA/加载脚本.py 输出的子集 JSON，按弱标签（答案别名出现在文档
标题或正文中）过滤出可评测样本，生成 framework demo 使用的 corpus.jsonl、
test.jsonl 与带哈希校验的 manifest.json。不调用 LLM，不做向量化。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT / "data" / "TriviaQA" / "outputs" / "wikipedia-dev_subset_100.json"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "data" / "demo"
DEFAULT_SEED = 20260829
DEFAULT_LIMIT = 20
DEMO_NAME = "triviaqa-demo-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="TriviaQA 子集 JSON 路径，默认 data/TriviaQA/outputs/wikipedia-dev_subset_100.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="demo 输出目录，默认 experiments/triviaqa/data/demo",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help="demo 题目数（弱标签过滤后取前 N 条），默认 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="确定性排序种子（仅用于 manifest 记录），默认 20260829",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return parsed


def load_subset(path: Path) -> dict[str, Any]:
    """读取子集 JSON 并校验 samples 列表存在。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if isinstance(samples, (str, bytes, bytearray)) or not isinstance(samples, Sequence):
        raise ValueError(f"Subset file must contain a 'samples' list: {path}")
    return payload


def build_records(
    samples: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """按弱标签过滤样本并构造共享语料与测试监督。"""
    labelled = [sample for sample in samples if _relevant_document_ids(sample)]
    if len(labelled) < limit:
        raise ValueError(
            f"Only {len(labelled)} questions have weak answer-alias labels, "
            f"expected at least {limit}"
        )

    corpus_by_id: dict[str, dict[str, Any]] = {}
    source_questions: dict[str, set[str]] = defaultdict(set)
    tests: list[dict[str, Any]] = []

    for sample in labelled[:limit]:
        question_id = str(sample["sample_id"])
        candidate_ids: list[str] = []
        for document in sample["documents"]:
            if not isinstance(document, Mapping):
                continue
            document_id = str(document.get("id") or "").strip()
            if not document_id:
                continue
            candidate_ids.append(document_id)
            text = str(document.get("text") or "").strip()
            title = str(document.get("title") or "").strip()
            existing = corpus_by_id.get(document_id)
            if existing is not None:
                if existing["text"] != text or existing["title"] != title:
                    raise ValueError(
                        f"Conflicting contexts found for document: {document_id}"
                    )
                source_questions[document_id].add(question_id)
                continue
            corpus_by_id[document_id] = {
                "id": document_id,
                "title": title,
                "text": text,
                "source": str(document.get("source") or "").strip(),
            }
            source_questions[document_id].add(question_id)

        aliases = [
            str(value).strip()
            for value in (sample.get("golden_answers") or ())
            if isinstance(value, str) and value.strip()
        ]
        relevant = _relevant_document_ids(sample)
        tests.append(
            {
                "id": question_id,
                "question": str(sample["query"]),
                "answer": str(sample.get("golden_answer") or "").strip(),
                "answers": aliases,
                "relevant_document_ids": relevant,
                "candidate_document_ids": candidate_ids,
            }
        )

    corpus = []
    for document_id in sorted(corpus_by_id):
        document = dict(corpus_by_id[document_id])
        document["source_question_ids"] = sorted(source_questions[document_id])
        corpus.append(document)
    weak_label_count = sum(len(test["relevant_document_ids"]) for test in tests)
    return corpus, sorted(tests, key=lambda record: record["id"]), weak_label_count


def _relevant_document_ids(sample: Mapping[str, Any]) -> list[str]:
    """返回标题或正文包含任一答案别名的文档 ID（与检索基准弱标签一致）。"""
    aliases = [
        _normalize_match(value)
        for value in (sample.get("golden_answers") or ())
        if str(value).strip() and _normalize_match(value) not in {"", "unk"}
    ]
    aliases = list(dict.fromkeys(aliases))
    relevant: list[str] = []
    for document in sample.get("documents") or ():
        if not isinstance(document, Mapping):
            continue
        evidence = (
            f" {_normalize_match(str(document.get('title') or '') + ' ' + str(document.get('text') or ''))} "
        )
        if any(f" {alias} " in evidence for alias in aliases):
            document_id = str(document.get("id") or "").strip()
            if document_id:
                relevant.append(document_id)
    return relevant


def _normalize_match(value: Any) -> str:
    """NFKC 规范化、小写并保留单词字符，与 experiments/retrieval/adapters 一致。"""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def write_demo(
    *,
    subset_path: Path,
    subset_payload: Mapping[str, Any],
    output: Path,
    seed: int,
    limit: int,
    corpus: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    weak_label_count: int,
) -> dict[str, Any]:
    """写入 JSONL 数据和包含校验摘要的 manifest。"""
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "corpus.jsonl"
    tests_path = output / "test.jsonl"
    _write_jsonl(corpus_path, corpus)
    _write_jsonl(tests_path, tests)

    manifest = {
        "schema_version": 1,
        "name": DEMO_NAME,
        "source": {
            "dataset": "TriviaQA",
            "subset_file": _repo_relative(subset_path),
            "subset_sha256": _sha256(subset_path),
            "subset_sampling": dict(subset_payload.get("sampling") or {}),
            "license": "research-only (no unified official license)",
        },
        "sampling": {
            "method": "weak-label-filtered-prefix",
            "seed": seed,
            "limit": limit,
        },
        "counts": {
            "documents": len(corpus),
            "test_examples": len(tests),
            "weak_labels": weak_label_count,
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


def build_demo(
    subset_path: Path,
    output: Path,
    *,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    """执行读取、弱标签过滤、语料构建和文件写入的完整流程。"""
    subset_payload = load_subset(subset_path)
    samples = subset_payload["samples"]
    corpus, tests, weak_label_count = build_records(samples, limit=limit)
    return write_demo(
        subset_path=subset_path,
        subset_payload=subset_payload,
        output=output,
        seed=seed,
        limit=limit,
        corpus=corpus,
        tests=tests,
        weak_label_count=weak_label_count,
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """以 UTF-8 JSON Lines 格式稳定写入记录。"""
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免一次读取大型子集。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    """返回相对仓库根目录的 posix 路径，仓库外路径原样返回。"""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行构建并输出不含样本正文的摘要。"""
    args = parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(
            f"TriviaQA subset file does not exist: {args.input}\n"
            "请先运行 data/TriviaQA/加载脚本.py 生成子集。"
        )
    manifest = build_demo(args.input, args.output, seed=args.seed, limit=args.limit)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
