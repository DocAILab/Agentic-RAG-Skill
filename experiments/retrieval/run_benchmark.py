"""`python -m experiments.retrieval.run_benchmark` 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .loading import iter_huggingface_items
from .retrievers import build_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run candidate-document retrieval evaluation")
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-config")
    parser.add_argument("--retriever", required=True, choices=("bm25", "vector"))
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir or (
        Path("experiments") / "retrieval" / "outputs" / args.dataset / args.retriever
    )
    metadata = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "retriever": args.retriever,
        "model": args.model if args.retriever == "vector" else None,
        "top_k": args.top_k,
    }
    retriever = build_retriever(
        args.retriever,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    items = iter_huggingface_items(
        args.dataset,
        args.split,
        config=args.dataset_config,
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


if __name__ == "__main__":
    raise SystemExit(main())
