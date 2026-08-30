"""Manifest helpers for FinanceBench local versions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class DatasetStateError(RuntimeError):
    """Raised when a local dataset version is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "records": jsonl_rows(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def existing_manifest(
    target: Path,
    expected_name: str,
    expected_revision: str,
) -> dict | None:
    if not target.exists():
        return None
    try:
        manifest = _read_manifest(target, expected_name, expected_revision)
        for record in manifest["files"]:
            _validate_file_record(target, record)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetStateError(
            f"Invalid local dataset at {target}; rerun with --force"
        ) from exc
    return manifest


def reset_target(target: Path, output_root: Path, force: bool) -> None:
    if not force or not target.exists():
        return
    if target.is_symlink() or target.parent.resolve() != output_root.resolve():
        raise DatasetStateError(f"Refusing to replace unsafe path: {target}")
    if target.name not in {"public", "small", "full"}:
        raise DatasetStateError(f"Refusing to replace unexpected version: {target}")
    shutil.rmtree(target)


def _read_manifest(target: Path, expected_name: str, revision: str) -> dict:
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("name") != expected_name:
        raise ValueError("unexpected dataset name")
    if manifest.get("source", {}).get("revision") != revision:
        raise ValueError("unexpected source revision")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("manifest contains no files")
    return manifest


def _validate_file_record(root: Path, record: dict) -> None:
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("manifest file is missing or outside its dataset directory")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"file size mismatch: {path}")
    if jsonl_rows(path) != record["records"]:
        raise ValueError(f"record count mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"checksum mismatch: {path}")
