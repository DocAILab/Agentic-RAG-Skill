"""`python -m experiments.retrieval.run_benchmark` 命令行入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .benchmark import run_benchmark
from .loading import DEFAULT_FINANCEBENCH_DEMO, iter_dataset_items
from .retrievers import (
    DEFAULT_BGE_BATCH_SIZE,
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_BM25_TITLE_B,
    DEFAULT_BM25_TITLE_BOOST,
    build_retriever,
    resolve_variant,
)
from .sampling import iter_manifest_items, read_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run candidate-document retrieval evaluation"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa", "financebench"),
    )
    parser.add_argument("--split")
    parser.add_argument("--dataset-config")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--retriever", required=True, choices=("bm25", "vector"))
    parser.add_argument("--variant")
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--title-b", type=float, default=DEFAULT_BM25_TITLE_B)
    parser.add_argument(
        "--title-boost", type=float, default=DEFAULT_BM25_TITLE_BOOST
    )
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BGE_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def build_run_metadata(args, *, code_commit: str | None = None) -> dict:
    variant = resolve_variant(args.retriever, args.variant)
    split = args.split or (
        "test" if args.dataset == "financebench" else "validation"
    )
    metadata = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": split,
        "retriever": args.retriever,
        "variant": variant,
        "model": args.model if args.retriever == "vector" else None,
        "top_k": args.top_k,
        "manifest": str(args.manifest) if args.manifest else None,
        "code_commit": code_commit or _current_commit(),
    }
    if args.dataset == "financebench":
        data_dir = args.data_dir or DEFAULT_FINANCEBENCH_DEMO
        metadata["data_dir"] = str(data_dir)
        demo_manifest = data_dir / "manifest.json"
        metadata["dataset_manifest_sha256"] = (
            _sha256(demo_manifest) if demo_manifest.is_file() else None
        )
    if args.manifest:
        metadata["manifest_digest"] = read_manifest(args.manifest)["digest"]
    if args.retriever == "bm25":
        parameters = {
            "k1": args.k1,
            "b": args.b,
        }
        if variant != "B0":
            parameters["title_boost"] = args.title_boost
        if variant in {"B2", "B3"}:
            parameters["title_b"] = args.b if args.title_b is None else args.title_b
        metadata["parameters"] = parameters
    else:
        metadata["representation"] = {
            "query": "raw" if variant == "V0" else "bge_instruction",
            "document_fields": (
                ["text"] if variant in {"V0", "V1"} else ["title", "text"]
            ),
        }
    return metadata


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.split is None:
        args.split = "test" if args.dataset == "financebench" else "validation"
    metadata = build_run_metadata(args)
    output = args.output_dir or (
        Path("experiments")
        / "retrieval"
        / "outputs"
        / args.dataset
        / metadata["variant"].lower()
    )
    retriever = build_retriever(
        args.retriever,
        variant=metadata["variant"],
        k1=args.k1,
        b=args.b,
        title_b=args.title_b,
        title_boost=args.title_boost,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    items = iter_dataset_items(
        args.dataset,
        args.split,
        config=args.dataset_config,
        data_dir=args.data_dir,
    )
    if args.manifest:
        manifest = read_manifest(args.manifest)
        items = iter_manifest_items(
            items,
            manifest,
            dataset=args.dataset,
            split=args.split,
        )
    summary = run_benchmark(
        items,
        retriever,
        output_dir=output,
        run_metadata=metadata,
        top_k=args.top_k,
        checkpoint_every=args.checkpoint_every,
        max_examples=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
