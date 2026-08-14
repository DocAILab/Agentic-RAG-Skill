"""JSONL 结果、checkpoint 和汇总文件持久化。"""

from __future__ import annotations

import json
from pathlib import Path


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_resume(record, run_metadata) -> None:
    expected = run_signature(run_metadata)
    if record.get("run_signature") != expected:
        raise ValueError("Existing results were produced with different run settings")


def run_signature(run_metadata) -> str:
    return json.dumps(run_metadata, ensure_ascii=False, sort_keys=True)
