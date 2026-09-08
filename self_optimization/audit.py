"""把每次 Skill 优化模型调用保存为可审计的独立文本和 JSON 文件。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import OptimizationProposal
from .optimizer import OptimizationModelCall


def write_optimization_audit(
    directory: str | Path,
    *,
    model_calls: Sequence[OptimizationModelCall],
    proposal: OptimizationProposal | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """持久化优化提示、原始响应、解析后提案和最终接收或拒绝结果。"""
    audit_dir = Path(directory).resolve()
    audit_dir.mkdir(parents=True, exist_ok=False)
    calls = tuple(model_calls)
    artifacts: dict[str, Any] = {
        "directory": str(audit_dir),
        "system_prompt_path": None,
        "prompt_path": None,
        "raw_response_path": None,
        "proposal_path": None,
        "result_path": str(audit_dir / "result.json"),
        "attempts": [],
    }
    attempts = []
    for index, model_call in enumerate(calls, start=1):
        attempt_dir = audit_dir / "attempts" / f"attempt-{index:04d}"
        attempt_dir.mkdir(parents=True)
        attempt_artifacts = _write_model_call(attempt_dir, model_call)
        attempts.append(
            {
                "attempt": index,
                "stage": model_call.stage,
                "error": model_call.error,
                "artifacts": attempt_artifacts,
            }
        )
    artifacts["attempts"] = attempts

    final_call = calls[-1] if calls else None
    call_metadata = _call_metadata(final_call)
    if final_call is not None:
        system_path = audit_dir / "system_prompt.txt"
        prompt_path = audit_dir / "prompt.txt"
        _write_text(system_path, final_call.system_prompt)
        _write_text(prompt_path, final_call.prompt)
        artifacts["system_prompt_path"] = str(system_path)
        artifacts["prompt_path"] = str(prompt_path)
        if final_call.raw_response is not None:
            response_path = audit_dir / "raw_response.txt"
            _write_text(response_path, final_call.raw_response)
            artifacts["raw_response_path"] = str(response_path)
    logged_proposal = proposal or (final_call.proposal if final_call is not None else None)
    if logged_proposal is not None:
        proposal_path = audit_dir / "proposal.json"
        _write_json(proposal_path, logged_proposal.to_dict())
        artifacts["proposal_path"] = str(proposal_path)
    _write_json(
        audit_dir / "result.json",
        {
            "schema_version": 1,
            "result": dict(result),
            "model_call": call_metadata,
            "attempt_count": len(calls),
            "attempts": attempts,
            "artifacts": artifacts,
        },
    )
    return dict(artifacts)


def compact_optimization_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """生成适合终端显示的优化摘要，完整审计路径和尝试细节仍保留在报告中。"""
    audit = result.get("audit")
    compact: dict[str, Any] = {
        "status": result.get("status"),
    }
    for key in ("selected_skill", "rationale", "reason", "error_type", "error"):
        if result.get(key) is not None:
            compact[key] = result[key]
    revision = result.get("revision")
    if isinstance(revision, Mapping):
        compact["revision_id"] = revision.get("revision_id")
        compact["changed_files"] = revision.get("changed_files")
        compact["diff_path"] = revision.get("diff_path")
    if isinstance(audit, Mapping):
        compact["optimizer_call_dir"] = audit.get("directory")
        attempts = audit.get("attempts")
        if isinstance(attempts, Sequence) and not isinstance(
            attempts,
            (str, bytes, bytearray),
        ):
            compact["model_call_count"] = len(attempts)
            compact["edit_attempt_count"] = sum(
                1
                for attempt in attempts
                if isinstance(attempt, Mapping) and attempt.get("stage") == "editing"
            )
    return compact


def _write_model_call(
    directory: Path,
    model_call: OptimizationModelCall,
) -> dict[str, Any]:
    """保存单次纠正尝试，并返回其文件路径和完整性元数据。"""
    system_path = directory / "system_prompt.txt"
    prompt_path = directory / "prompt.txt"
    _write_text(system_path, model_call.system_prompt)
    _write_text(prompt_path, model_call.prompt)
    artifacts: dict[str, Any] = {
        "directory": str(directory),
        "stage": model_call.stage,
        "system_prompt_path": str(system_path),
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "proposal_path": None,
        **_call_metadata(model_call),
    }
    if model_call.raw_response is not None:
        response_path = directory / "raw_response.txt"
        _write_text(response_path, model_call.raw_response)
        artifacts["raw_response_path"] = str(response_path)
    if model_call.proposal is not None:
        proposal_path = directory / "proposal.json"
        _write_json(proposal_path, model_call.proposal.to_dict())
        artifacts["proposal_path"] = str(proposal_path)
    return artifacts


def _call_metadata(model_call: OptimizationModelCall | None) -> dict[str, Any]:
    """生成单次模型调用的字符数和内容哈希，不在元数据中重复大段文本。"""
    if model_call is None:
        return {
            "prompt_characters": 0,
            "response_characters": 0,
            "prompt_sha256": None,
            "response_sha256": None,
        }
    response = model_call.raw_response
    return {
        "prompt_characters": len(model_call.prompt),
        "response_characters": len(response) if response is not None else 0,
        "prompt_sha256": _text_hash(model_call.prompt),
        "response_sha256": _text_hash(response) if response is not None else None,
    }


def _write_text(path: Path, content: str) -> None:
    """以稳定 UTF-8 和 LF 换行写入完整模型文本。"""
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以便于人工检查的缩进格式写入审计 JSON。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _text_hash(content: str) -> str:
    """计算模型文本的 UTF-8 SHA-256 以便核对日志完整性。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
