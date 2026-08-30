"""Create a deterministic tuning or screening sample manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loading import iter_huggingface_items
from .sampling import build_manifest, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a retrieval sample manifest")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa", "financebench"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-config")
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    items = iter_huggingface_items(
        args.dataset,
        args.split,
        config=args.dataset_config,
    )
    manifest = build_manifest(
        items,
        dataset=args.dataset,
        split=args.split,
        size=args.size,
    )
    write_manifest(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
