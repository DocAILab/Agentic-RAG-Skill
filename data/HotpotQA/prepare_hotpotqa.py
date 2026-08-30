"""Command-line interface for preparing reproducible HotpotQA datasets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from hotpotqa_data import prepare_versions


DATA_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=("full", "small", "all"),
        default="all",
        help="Dataset version to materialize (default: all).",
    )
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = prepare_versions(
        args.version,
        args.output_root,
        force=args.force,
    )
    summary = {
        name: {
            "files": len(manifest["files"]),
            "records": sum(record["records"] for record in manifest["files"]),
        }
        for name, manifest in manifests.items()
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
