"""Compare two paired retrieval result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import compare_paired_metric
from .persistence import iter_jsonl, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare paired retrieval runs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", action="append")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    baseline = list(iter_jsonl(args.baseline))
    candidate = list(iter_jsonl(args.candidate))
    metrics = args.metric or ["all_support@5", "recall@5", "mrr"]
    payload = {
        "dataset": args.dataset,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "comparisons": {
            metric: compare_paired_metric(
                baseline,
                candidate,
                metric,
                resamples=args.resamples,
            )
            for metric in metrics
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
