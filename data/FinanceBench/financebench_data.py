"""Materialize reproducible public and small FinanceBench datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from _manifest import (
    DatasetStateError,
    existing_manifest,
    file_record,
    reset_target,
    write_manifest,
)

REPOSITORY = "PatronusAI/financebench"
REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"
DEFAULT_SPLIT = "train"
PUBLIC_NAME = "financebench-public-v1"
SMALL_NAME = "financebench-small-v1"
SMALL_SALT = "FinanceBench-small-v1"
DEFAULT_SMALL_SIZE = 150


def load_financebench_rows(
    *,
    dataset_name: str = REPOSITORY,
    split: str = DEFAULT_SPLIT,
    revision: str = REVISION,
    load_dataset_fn=None,
) -> list[Mapping[str, Any]]:
    loader = load_dataset_fn or _load_dataset
    rows = loader(dataset_name, split=split, revision=revision, streaming=True)
    return [dict(row) for row in rows]


def prepare_versions(
    version: str,
    output_root: Path,
    *,
    force: bool = False,
    load_dataset_fn=None,
    small_size: int = DEFAULT_SMALL_SIZE,
) -> dict[str, dict]:
    if version not in {"public", "small", "all"}:
        raise ValueError(f"Unsupported FinanceBench version: {version}")
    if small_size < 1:
        raise ValueError("small_size must be positive")

    requested = ("public", "small") if version == "all" else (version,)
    results: dict[str, dict] = {}

    if not force:
        for name in requested:
            target = Path(output_root) / name
            expected = _source(name, small_size=small_size)
            existing = (
                existing_manifest(
                    target,
                    PUBLIC_NAME if name == "public" else SMALL_NAME,
                    REVISION,
                )
                if target.exists()
                else None
            )
            if existing is not None:
                if existing.get("source") != expected:
                    raise DatasetStateError(
                        f"Existing {name} dataset uses different source or sampling "
                        "settings; rerun with --force"
                    )
                results[name] = existing
    if len(results) == len(requested):
        return results

    rows = load_financebench_rows(load_dataset_fn=load_dataset_fn)

    if version in {"public", "all"}:
        results["public"] = _materialize(
            rows,
            output_root,
            "public",
            force,
            source=_source("public", small_size=small_size),
        )
    if version in {"small", "all"}:
        selected = _stable_sample(rows, small_size)
        results["small"] = _materialize(
            selected,
            output_root,
            "small",
            force,
            source={
                "repository": REPOSITORY,
                "revision": REVISION,
                "split": DEFAULT_SPLIT,
                "sampling": {
                    "method": "sha256-id-ranking",
                    "salt": SMALL_SALT,
                    "size": small_size,
                },
            },
        )
    return results


def _materialize(
    rows: Iterable[Mapping[str, Any]],
    root: Path,
    name: str,
    force: bool,
    *,
    source: dict,
) -> dict:
    records = [dict(row) for row in rows]
    target = Path(root) / name
    reset_target(target, Path(root), force)
    existing = existing_manifest(
        target,
        PUBLIC_NAME if name == "public" else SMALL_NAME,
        REVISION,
    )
    if existing is not None:
        if existing.get("source") != source:
            raise DatasetStateError(
                f"Existing {name} dataset uses different source or sampling "
                "settings; rerun with --force"
            )
        return existing
    target.mkdir(parents=True, exist_ok=False)
    raw_path = target / f"{DEFAULT_SPLIT}.jsonl"
    _write(raw_path, records)
    manifest = {
        "schema_version": 1,
        "name": PUBLIC_NAME if name == "public" else SMALL_NAME,
        "source": source,
        "storage": "raw-jsonl",
        "counts": {
            "examples": len(records),
            "evidence_items": _evidence_count(records),
        },
        "files": [file_record(raw_path, target)],
    }
    write_manifest(target, manifest)
    return manifest


def _stable_sample(
    rows: Iterable[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("small_size must be positive")
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            hashlib.sha256(
                f"{SMALL_SALT}:{_row_id(item[1], item[0])}".encode()
            ).hexdigest(),
            _row_id(item[1], item[0]),
        ),
    )
    if len(ranked) < limit:
        raise ValueError(
            f"small_size {limit} exceeds the {len(ranked)} available examples"
        )
    return [dict(row) for _, row in ranked[:limit]]


def _source(name: str, *, small_size: int) -> dict[str, Any]:
    source: dict[str, Any] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "split": DEFAULT_SPLIT,
    }
    if name == "small":
        source["sampling"] = {
            "method": "sha256-id-ranking",
            "salt": SMALL_SALT,
            "size": small_size,
        }
    return source


def _row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("financebench_id", "id", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(index)


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(dict(record), ensure_ascii=False) for record in rows)
        + "\n",
        encoding="utf-8",
    )


def _evidence_count(rows: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        evidence = row.get("evidence")
        if isinstance(evidence, list):
            total += len(evidence)
        elif evidence:
            total += 1
    return total


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "FinanceBench preparation requires the 'datasets' package"
        ) from exc
    return load_dataset(*args, **kwargs)
