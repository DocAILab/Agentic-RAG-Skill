"""Prepare reproducible local TriviaQA text subsets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from triviaqa_data import (
    DEFAULT_EVIDENCE,
    DEFAULT_MAX_QUERY_SAMPLES,
    DEFAULT_OUTPUT,
    DEFAULT_SEED,
    DEFAULT_SOURCE,
    prepare_subsets,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="TriviaQA 问答 JSON(.gz) 文件路径，默认 data/raw/triviaqa/qa/wikipedia-dev.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="证据根目录，需包含 wikipedia/ 与 web/ 子目录，默认 data/raw/triviaqa/evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="子集 JSON 与 manifest 输出目录，默认 data/TriviaQA/outputs",
    )
    parser.add_argument(
        "--max-query-samples",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_MAX_QUERY_SAMPLES),
        dest="max_query_samples",
        metavar="N",
        help="一个或多个裁剪子集规模，例如 --max-query-samples 100 800",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="确定性排序种子，默认 20260828",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略现有 manifest 校验并强制重建请求的所有子集",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    """把命令行字符串解析为正整数。"""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max query samples must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_subsets(
        source=args.source,
        evidence_root=args.evidence,
        output_root=args.output,
        sizes=args.max_query_samples,
        seed=args.seed,
        force=args.force,
    )
    summary = {
        "name": manifest["name"],
        "total_valid_samples": manifest["counts"]["samples"],
        "filtered": manifest["counts"]["filtered"],
        "files": [
            {
                "path": record["path"],
                "records": record["records"],
                "bytes": record["bytes"],
            }
            for record in manifest["files"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
