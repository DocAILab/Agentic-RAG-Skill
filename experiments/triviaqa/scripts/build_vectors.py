"""从 TriviaQA 文本子集构建稠密向量索引。

读取 data/TriviaQA/加载脚本.py 输出的标准化 JSON 子集，把其中的全部文档去重、
可选按 token 切块后编码为稠密向量，并持久化为 framework Vector Retriever 可直接
复用的 manifest.json + vectors.npy 索引。本脚本只做向量化与索引持久化，不执行
检索、不调用 LLM、不生成答案。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from framework.models import SentenceTransformerEmbeddingClient
    from framework.vector_index import (
        DenseVectorIndex,
        build_or_load_vector_index,
        embedding_model_fingerprint,
    )
except ImportError:
    # 直接以脚本方式运行（python -B experiments/triviaqa/scripts/build_vectors.py）
    # 时，把仓库根目录补进 sys.path，保证 framework 包可导入。
    SCRIPT_DIR_FALLBACK = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR_FALLBACK.parents[3]))
    from framework.models import SentenceTransformerEmbeddingClient
    from framework.vector_index import (
        DenseVectorIndex,
        build_or_load_vector_index,
        embedding_model_fingerprint,
    )

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "TriviaQA" / "outputs"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "TriviaQA" / "outputs" / "vector_index"
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

# 与 framework/skills/components/component-vector-retriever/scripts/component.py
# 保持一致，保证缓存键与框架运行时完全兼容。
TEXT_FORMAT_VERSION = "component-vector-retriever:title-text:v1"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


def parse_args() -> argparse.Namespace:
    """解析子集输入、索引输出、模型与分块参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="子集 JSON 文件或目录；目录将按名称扫描全部 *_subset_*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="向量索引输出根目录（写入 <cache_key>/manifest.json + vectors.npy 与 corpus.jsonl）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="sentence-transformers 模型名，默认 BAAI/bge-large-en-v1.5",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch 设备：auto/cpu/cuda，默认自动选择",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        metavar="N",
        help="embedding 批大小，默认 32",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=_non_negative_int,
        default=512,
        metavar="N",
        help="按 token 切块上限，默认 512；设为 0 表示整篇不切块（会被模型截断）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="构建完成后对前 3 条 query 做检索自检并打印命中结果",
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    """把命令行字符串解析为正整数。"""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    """把命令行字符串解析为非负整数。"""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def load_samples(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """读取一个或多个子集 JSON，返回全部样本。"""
    samples: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("samples") if isinstance(payload, Mapping) else None
        if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
            raise ValueError(f"Subset file must contain a 'samples' list: {path}")
        samples.extend(
            dict(item) for item in items if isinstance(item, Mapping)
        )
    if not samples:
        raise ValueError("No samples found in the input subset files")
    return samples


def collect_documents(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """收集全部文档，按文档 ID 去重并按 ID 确定性排序。"""
    documents: dict[str, dict[str, Any]] = {}
    for sample in samples:
        items = sample.get("documents")
        if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            document_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not document_id or not text:
                continue
            documents.setdefault(
                document_id,
                {
                    "id": document_id,
                    "title": str(item.get("title") or "").strip(),
                    "text": text,
                    "source": str(item.get("source") or "").strip(),
                },
            )
    if not documents:
        raise ValueError("No documents found in the input subset samples")
    return [documents[key] for key in sorted(documents)]


def chunk_documents(
    documents: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chunk_tokens: int,
) -> list[dict[str, Any]]:
    """按 token 窗口把长文档切成多个 chunk，chunk ID 为 {doc_id}#{序号}。"""
    if chunk_tokens <= 0:
        return [dict(document) for document in documents]
    chunked: list[dict[str, Any]] = []
    for document in documents:
        text = str(document["text"])
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for offset in range(0, len(token_ids), chunk_tokens):
            window = token_ids[offset : offset + chunk_tokens]
            pieces = tokenizer.convert_ids_to_tokens(window)
            start = 0
            while start < len(pieces) and pieces[start].startswith("##"):
                start += 1
            pieces = pieces[start:]
            if not pieces:
                continue
            chunk_text = tokenizer.convert_tokens_to_string(pieces).strip()
            if not chunk_text:
                continue
            chunked.append(
                {
                    **dict(document),
                    "id": f"{document['id']}#{offset // chunk_tokens:04d}",
                    "text": chunk_text,
                    "parent_id": str(document["id"]),
                }
            )
    return chunked


def document_text(document: Mapping[str, Any]) -> str:
    """把标题与正文组合为与 Vector Retriever 一致的文本格式。"""
    parts = [str(document.get("title") or "").strip(), str(document.get("text") or "")]
    return "\n".join(part for part in parts if part)


def make_embed(
    client: SentenceTransformerEmbeddingClient,
    batch_size: int,
) -> Any:
    """包装 embedding 客户端，分批编码并打印进度。"""

    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        encoded: list[list[float]] = []
        total = len(texts)
        next_mark = 500
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            encoded.extend(client.embed(texts[start:end]))
            if end >= next_mark or end == total:
                print(f"  embedded {end}/{total} documents", flush=True)
                next_mark = ((end // 500) + 1) * 500
        return encoded

    return embed


def build_index(
    documents: Sequence[Mapping[str, Any]],
    *,
    client: SentenceTransformerEmbeddingClient,
    batch_size: int,
    cache_root: Path,
) -> DenseVectorIndex:
    """复用 framework 构建或加载向量索引，并原子写盘。"""
    document_ids = [str(document["id"]) for document in documents]
    document_texts = [document_text(document) for document in documents]
    return build_or_load_vector_index(
        document_ids,
        document_texts,
        embed=make_embed(client, batch_size),
        embedding_fingerprint=embedding_model_fingerprint(client),
        text_format_version=TEXT_FORMAT_VERSION,
        cache_root=cache_root,
    )


def write_corpus(path: Path, documents: Sequence[Mapping[str, Any]]) -> None:
    """把去重/切块后的文档写入 JSON Lines 语料文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(document), ensure_ascii=False, sort_keys=True) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def verify_index(
    index: DenseVectorIndex,
    client: SentenceTransformerEmbeddingClient,
    samples: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> None:
    """对前几条真实 query 做一次检索自检，验证索引可用。"""
    print("self_check:")
    for sample in samples[:3]:
        query = str(sample.get("query") or "").strip()
        if not query:
            continue
        query_text = f"{DEFAULT_QUERY_INSTRUCTION} {query}".strip()
        vector = client.embed([query_text])[0]
        hits = index.search(vector, top_k)
        print(f"  query: {query[:70]}")
        for position, (doc_index, score) in enumerate(hits, start=1):
            print(
                f"    top{position}: {index.document_ids[doc_index]} "
                f"score={score:.4f}"
            )


def _repo_relative(path: Path) -> str:
    """返回相对仓库根目录的 posix 路径，仓库外路径原样返回。"""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    """执行读取、去重、切块、向量化与自检。"""
    args = parse_args()
    if args.input.is_dir():
        paths = sorted(args.input.glob("*_subset_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No *_subset_*.json files found in input directory: {args.input}"
            )
    elif args.input.is_file():
        paths = [args.input]
    else:
        raise FileNotFoundError(
            f"Input does not exist (expected a subset JSON or directory): {args.input}"
        )
    print("subset_files:", [_repo_relative(path) for path in paths])

    samples = load_samples(paths)
    documents = collect_documents(samples)
    print(f"unique_documents: {len(documents)}")

    client = SentenceTransformerEmbeddingClient(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    client.load()
    if args.chunk_tokens > 0:
        documents = chunk_documents(
            documents,
            tokenizer=client.encoder.tokenizer,
            chunk_tokens=args.chunk_tokens,
        )
        print(
            f"chunked_documents: {len(documents)} "
            f"(chunk_tokens={args.chunk_tokens})"
        )

    index = build_index(
        documents,
        client=client,
        batch_size=args.batch_size,
        cache_root=args.output,
    )
    corpus_path = args.output / "corpus.jsonl"
    write_corpus(corpus_path, documents)
    if args.verify:
        verify_index(index, client, samples)

    print(
        json.dumps(
            {
                "cache_key": index.cache_key,
                "dimension": index.dimension,
                "documents": len(index.document_ids),
                "source": index.source,
                "index_dir": (
                    _repo_relative(index.cache_path)
                    if index.cache_path is not None
                    else None
                ),
                "corpus_file": _repo_relative(corpus_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
