"""Materialize reproducible full and small local HotpotQA datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from _manifest import (
    DatasetStateError,
    existing_manifest,
    file_record,
    reset_target,
    sha256_file,
    write_manifest,
)

__all__ = [
    "DatasetStateError",
    "REPOSITORY",
    "REVISION",
    "prepare_full",
    "prepare_small",
    "prepare_versions",
    "select_small_indices",
    "sha256_file",
]


REPOSITORY = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SMALL_SALT = "HotpotQA-small-v1"
SMALL_SIZE = 5_000
CONFIG_SPLITS = {
    "distractor": ("train", "validation"),
    "fullwiki": ("train", "validation", "test"),
}
REQUIRED_COLUMNS = {
    "id",
    "question",
    "answer",
    "type",
    "level",
    "supporting_facts",
    "context",
}


def select_small_indices(ids: Sequence[str], *, size: int = SMALL_SIZE) -> list[int]:
    """Select a source-order-stable subset using salted ID hashes."""
    if size < 1 or len(ids) < size:
        raise ValueError(f"small size must be between 1 and {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("HotpotQA IDs must be unique")
    ranked = sorted(
        enumerate(ids),
        key=lambda item: (
            hashlib.sha256(f"{SMALL_SALT}:{item[1]}".encode()).hexdigest(),
            item[1],
        ),
    )
    return sorted(index for index, _ in ranked[:size])


def prepare_small(
    output_root: Path,
    *,
    force: bool = False,
    load_dataset_fn=None,
) -> dict:
    """Materialize the fixed 5,000-row HotpotQA training subset."""
    target = Path(output_root) / "small"
    reset_target(target, Path(output_root), force)
    existing = existing_manifest(target, "hotpotqa-small-v1", REVISION)
    if existing is not None:
        return existing
    loader = load_dataset_fn or _load_dataset
    source = loader(
        REPOSITORY,
        "distractor",
        split="train",
        revision=REVISION,
    )
    _validate_columns(source, "distractor/train")
    indices = select_small_indices([str(value) for value in source["id"]])
    target.mkdir(parents=True, exist_ok=False)
    parquet_path = target / "train-5000.parquet"
    source.select(indices).to_parquet(parquet_path)
    manifest = _small_manifest(parquet_path, target)
    write_manifest(target, manifest)
    return manifest


def prepare_full(
    output_root: Path,
    *,
    force: bool = False,
    load_dataset_fn=None,
) -> dict:
    """Materialize every official HotpotQA configuration and split."""
    target = Path(output_root) / "full"
    reset_target(target, Path(output_root), force)
    existing = existing_manifest(target, "hotpotqa-full-v1", REVISION)
    if existing is not None:
        return existing
    loader = load_dataset_fn or _load_dataset
    target.mkdir(parents=True, exist_ok=False)
    files = []
    for configuration, expected_splits in CONFIG_SPLITS.items():
        dataset = loader(REPOSITORY, configuration, revision=REVISION)
        _validate_splits(dataset, configuration, expected_splits)
        config_root = target / configuration
        config_root.mkdir()
        for split in expected_splits:
            _validate_columns(dataset[split], f"{configuration}/{split}")
            path = config_root / f"{split}.parquet"
            dataset[split].to_parquet(path)
            files.append(file_record(path, target))
    manifest = _full_manifest(files)
    write_manifest(target, manifest)
    return manifest


def prepare_versions(
    version: str,
    output_root: Path,
    *,
    force: bool = False,
    load_dataset_fn=None,
) -> dict[str, dict]:
    names = ("full", "small") if version == "all" else (version,)
    builders = {"full": prepare_full, "small": prepare_small}
    if any(name not in builders for name in names):
        raise ValueError(f"Unsupported HotpotQA version: {version}")
    return {
        name: builders[name](
            output_root,
            force=force,
            load_dataset_fn=load_dataset_fn,
        )
        for name in names
    }


def _small_manifest(path: Path, root: Path) -> dict:
    return {
        "schema_version": 1,
        "name": "hotpotqa-small-v1",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "configuration": "distractor",
            "split": "train",
        },
        "sampling": {
            "method": "sha256-id-ranking",
            "salt": SMALL_SALT,
            "size": SMALL_SIZE,
        },
        "files": [file_record(path, root)],
    }


def _full_manifest(files: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "name": "hotpotqa-full-v1",
        "source": {"repository": REPOSITORY, "revision": REVISION},
        "configurations": {
            name: list(splits) for name, splits in CONFIG_SPLITS.items()
        },
        "files": files,
    }


def _validate_splits(dataset, configuration: str, expected: Sequence[str]) -> None:
    missing = set(expected) - set(dataset)
    if missing:
        raise ValueError(f"{configuration} is missing splits: {sorted(missing)}")


def _validate_columns(dataset, label: str) -> None:
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HotpotQA preparation requires the project experiment dependencies"
        ) from exc
    return load_dataset(*args, **kwargs)
