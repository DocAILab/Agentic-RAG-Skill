"""Deterministic sample manifests for retrieval tuning and screening."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from .loading import DatasetItem
from .persistence import write_json


def build_manifest(
    items: Iterable[DatasetItem],
    *,
    dataset: str,
    split: str,
    size: int,
) -> dict:
    """Select the valid examples with the lowest stable SHA-256 keys."""
    if size < 1:
        raise ValueError("manifest size must be positive")
    selected = []
    seen = set()
    for item in items:
        if item.error or item.example is None:
            continue
        if item.sample_id in seen:
            raise ValueError(f"duplicate sample id: {item.sample_id}")
        seen.add(item.sample_id)
        digest = _sample_digest(dataset, item.sample_id)
        entry = (-int(digest, 16), item.sample_id, digest)
        if len(selected) < size:
            heapq.heappush(selected, entry)
        elif entry[0] > selected[0][0]:
            heapq.heapreplace(selected, entry)
    if len(selected) != size:
        raise ValueError(f"requested {size} valid examples, found {len(selected)}")
    selected_ids = [item[1] for item in sorted(selected, key=lambda item: item[2])]
    payload = {
        "dataset": dataset,
        "split": split,
        "requested_size": size,
        "selected_ids": selected_ids,
    }
    return {**payload, "digest": _manifest_digest(payload)}


def write_manifest(path: str | Path, manifest: Mapping) -> None:
    target = Path(path)
    if target.is_file():
        existing = read_manifest(target)
        if existing != dict(manifest):
            raise ValueError("refusing to overwrite a different manifest")
        return
    write_json(target, dict(manifest))


def read_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = payload.pop("digest", None)
    if digest != _manifest_digest(payload):
        raise ValueError("manifest digest does not match its contents")
    return {**payload, "digest": digest}


def iter_manifest_items(
    items: Iterable[DatasetItem], manifest: Mapping, *, dataset: str, split: str
) -> Iterator[DatasetItem]:
    if manifest.get("dataset") != dataset or manifest.get("split") != split:
        raise ValueError("manifest dataset or split does not match the run")
    selected = set(manifest.get("selected_ids", ()))
    found = set()
    for item in items:
        if item.sample_id in selected:
            found.add(item.sample_id)
            yield item
    missing = selected - found
    if missing:
        raise ValueError(f"manifest samples were not found: {sorted(missing)[:3]}")


def _sample_digest(dataset: str, sample_id: str) -> str:
    value = f"{dataset}:{sample_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _manifest_digest(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
