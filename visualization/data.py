"""Read-only, dependency-free views over recorded XRAGOS run artifacts.

No experiment modules are imported: opening the visualizer cannot initialize a
model, change a checkpoint, or make an API request. Prompt bodies are read only
when a node is opened. Files are selected by recorded artifacts, never by an
HTTP-supplied path.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NODES = {
    "input": "任务输入", "manage": "Manage 规划", "agentic": "Agentic 选择",
    "components": "Component 绑定", "execution": "工作流执行",
    "evaluation": "效果评估", "policy_select": "Policy 选择优化对象",
    "policy_edit": "Policy 修改 Skill", "validation": "执行验证",
    "inheritance": "Skill 更新与继承",
}
STAGE_NODES = {"manage": "manage", "agentic": "agentic", "component": "components",
               "generation": "execution", "policy_selection": "policy_select",
               "policy_editing": "policy_edit"}
_SECRET_KEY = re.compile(r"^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
                         r"authorization|password|passwd|secret|client[_-]?secret|"
                         r"credentials|private[_-]?key)$", re.I)
_TOKEN = re.compile(r"\bsk-[A-Za-z0-9._-]{8,}\b")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}", re.I)
_ASSIGNMENT = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
    r"[\"']?\s*[:=]\s*[\"'])([^\"'\r\n]+)")
_SUFFIXES = {".json", ".jsonl", ".txt", ".py", ".md", ".yaml", ".yml", ".diff"}


def redact(value: Any) -> Any:
    """Remove credentials without loading or retaining a credential dictionary."""
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if _SECRET_KEY.match(str(k)) else redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _ASSIGNMENT.sub(r"\1[REDACTED]", _BEARER.sub("Bearer [REDACTED]",
                                                         _TOKEN.sub("[REDACTED]", value)))
    return value


def _blocked(path: Path) -> bool:
    for part in path.parts:
        name = part.lower()
        if (name == ".git" or name.startswith(".env") or name.startswith("settings")
                or name.startswith("secrets") or name.startswith("credentials")
                or name in {"secret", "api_key", "apikey", "id_rsa", "id_ed25519"}):
            return True
    return path.suffix.lower() not in _SUFFIXES


def _within(path: Path, root: Path) -> bool:
    try:
        _path(path).relative_to(_path(root))
        return True
    except ValueError:
        return False


def _path(value: Any) -> Path:
    # Windows extended-length paths are emitted by the experiment runner.
    raw = str(value)
    if raw.startswith("\\\\?\\"):
        raw = raw[4:]
    return Path(raw)


def _json(path: Path, default: Any = None) -> Any:
    path = path.resolve()
    if _blocked(path):
        return default
    try:
        if path.stat().st_size > 64 * 1024 * 1024:
            return default
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return default


def _jsonl(path: Path) -> list[dict]:
    """A writer may be between writes; malformed/incomplete lines stay invisible."""
    path = path.resolve()
    if _blocked(path):
        return []
    rows = []
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, UnicodeError):
        pass
    return rows


def _children(root: Path) -> list[Path]:
    try:
        return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return []


@dataclass(frozen=True)
class _Run:
    root: Path
    batch: Path

    @property
    def id(self) -> str:
        return hashlib.sha256(str(self.root).casefold().encode()).hexdigest()[:20]


class RunStore:
    """Discover bounded run locations and provide normalized, redacted snapshots."""

    def __init__(self, runtime_root: Path, repo_root: Path):
        self.runtime_root = _path(runtime_root).resolve()
        self.repo_root = _path(repo_root).resolve()
        self._summaries: dict[str, tuple[tuple, dict]] = {}

    @staticmethod
    def _is_run(path: Path) -> bool:
        return ((path / "task-flow.report.json").is_file()
                or (path / "task-flow.events.jsonl").is_file())

    def _discover(self) -> list[_Run]:
        found: dict[Path, _Run] = {}

        def add(path: Path, batch: Path):
            resolved = path.resolve()
            if _within(resolved, self.runtime_root) and self._is_run(resolved):
                found[resolved] = _Run(resolved, batch.resolve())

        add(self.runtime_root, self.runtime_root)
        for run in _children(self.runtime_root / "runs"):
            add(run, self.runtime_root)
        for batch in _children(self.runtime_root):
            # Do not recurse into environments, vector indexes, caches or tests.
            if batch.name == "runs":
                continue
            add(batch, batch)
            if batch.name.startswith("real-"):
                for run in _children(batch / "runs"):
                    add(run, batch)
        return sorted(found.values(), key=lambda r: str(r.root), reverse=True)

    def _find(self, run_id: str) -> _Run:
        for ref in self._discover():
            if ref.id == run_id:
                return ref
        raise KeyError("运行不存在")

    def _artifact_path(self, ref: _Run, value: Any) -> Path | None:
        if not value:
            return None
        path = _path(value)
        if not path.is_absolute():
            path = ref.root / path
        path = path.resolve()
        if _blocked(path) or not (_within(path, ref.root) or _within(path, ref.batch)):
            return None
        return path

    def _read_text(self, ref: _Run, value: Any) -> tuple[str | None, str | None]:
        path = self._artifact_path(ref, value)
        if path is None:
            return None, "记录文件路径未获准读取"
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                return None, "记录文件超过 8 MiB 展示上限"
            return path.read_text(encoding="utf-8-sig"), None
        except (OSError, UnicodeError):
            return None, "记录文件缺失或尚未写完"

    def _artifact_json(self, ref: _Run, value: Any) -> Any:
        path = self._artifact_path(ref, value)
        return _json(path) if path else None

    @staticmethod
    def _call_node(call: dict) -> str | None:
        if call.get("scope") == "revision_validation":
            return "validation"
        return STAGE_NODES.get(call.get("stage"))

    @staticmethod
    def _same_attempt(row: dict, task: dict) -> bool:
        """Missing attempt identity must not be attached to a known retry."""
        current = task.get("attempt_id")
        return row.get("attempt_id") == current if current else row.get("attempt_id") is None

    def _calls(self, ref: _Run, report: dict, start: dict) -> tuple[list[dict], dict[str, dict], dict[str, list], int]:
        captures: dict[str, dict] = {}
        capture_root = ref.root / "model-captures"
        if not (capture_root / "capture-log.jsonl").is_file():
            capture_root = ref.batch
        for row in self._rows(ref, capture_root / "capture-log.jsonl"):
            if row.get("generate_id"):
                captures[str(row["generate_id"])] = row
        logs: dict[str, list] = {}
        log_root = capture_root if (capture_root / "api-calls.jsonl").is_file() else ref.batch
        for row in self._rows(ref, log_root / "api-calls.jsonl"):
            if row.get("generate_id"):
                logs.setdefault(str(row["generate_id"]), []).append(row)
        calls = []
        unresolved = 0
        group_id = report.get("group", {}).get("group_id") or start.get("group_id")
        run_name = report.get("run_id") or ref.root.name
        scoped = capture_root == ref.root / "model-captures"
        shared = len(_children(ref.batch / "runs")) > 1
        kept = set()
        for cid in dict.fromkeys([*captures, *logs]):
            cap = captures.get(cid, {})
            rows = logs.get(cid, [])
            merged = {}
            for row in rows:
                merged.update(row)
            merged.update(cap)
            groups = {r["group_id"] for r in rows + [cap] if r.get("group_id")}
            run_ids = {r["run_id"] for r in rows + [cap] if r.get("run_id")}
            if len(groups) > 1 or len(run_ids) > 1:
                unresolved += 1
                continue
            if groups and group_id and group_id not in groups:
                continue
            if run_ids and run_name not in run_ids:
                continue
            if not scoped and shared and not ((groups and group_id in groups) or run_name in run_ids):
                unresolved += 1
                continue
            kept.add(cid)
            if cap:
                state = cap.get("status", "unknown")
            elif any(r.get("event") == "generate_completed" for r in rows):
                state = "completed"
            elif any(r.get("event") == "generate_failed" for r in rows):
                state = "failed"
            elif rows:
                state = "running"
            else:
                state = "unknown"
            transports = [r for r in rows if r.get("event") == "transport_started"]
            statistics = merged.get("request_statistics") if merged.get("request_statistics_availability") == "client_reported_last_call" else None
            statistics = statistics if isinstance(statistics, dict) else {}
            last_attempts = statistics.get("attempts")
            last_retries = statistics.get("retries")
            last_attempts = last_attempts if type(last_attempts) is int and last_attempts >= 0 else None
            last_retries = last_retries if type(last_retries) is int and last_retries >= 0 else None
            http_attempts = len(transports) if transports else last_attempts
            http_retries = max(len(transports) - 1, 0) if transports else last_retries
            calls.append({"id": cid, "role": merged.get("role"), "model": merged.get("model"),
                          "stage": merged.get("stage", "unknown"), "scope": merged.get("scope", "unknown"),
                          "task_id": merged.get("task_id"), "question_id": merged.get("question_id"),
                          "attempt_id": merged.get("attempt_id"), "status": state,
                          "seconds": merged.get("seconds"), "timestamp": merged.get("timestamp"),
                          "skill_hash": merged.get("skill_hash"),
                          "component": merged.get("component_name") or merged.get("component"),
                          "slot": merged.get("slot"),
                          "component_call_id": merged.get("component_call_id"),
                          "prompt_characters": merged.get("prompt_characters"),
                          "response_characters": merged.get("response_characters"),
                          "capture_available": bool(cap),
                          "http_attempt_count": http_attempts,
                          "provider": merged.get("provider"), "request_statistics": merged.get("request_statistics"),
                          "request_statistics_availability": merged.get("request_statistics_availability", "unknown"),
                          "parameters": merged.get("parameters"),
                          "parameters_scope": merged.get("parameters_scope", "unknown"),
                          "http_statistics_source": "transport_events" if transports else "client_reported_last_call" if statistics else "unknown",
                          "text_not_recorded_reason": merged.get("text_not_recorded_reason"),
                          "http_retry_count": http_retries,
                          "http_failure_count": sum(r.get("event") == "transport_failed" for r in rows) if transports else None,
                          "recovered_after_retry": (state == "completed" and any(r.get("event") == "transport_failed" for r in rows)) if transports else (state == "completed" and last_retries > 0) if last_retries is not None else None,
                          "node_id": self._call_node(merged)})
        calls.sort(key=lambda c: (c.get("timestamp") or "", c["id"]))
        return calls, {k: v for k, v in captures.items() if k in kept}, {k: v for k, v in logs.items() if k in kept}, unresolved

    def _rows(self, ref: _Run, path: Path) -> list[dict]:
        allowed = self._artifact_path(ref, path)
        return _jsonl(allowed) if allowed else []

    def _tail_rows(self, ref: _Run, path: Path) -> list[dict]:
        """Bounded event tail for list status, without decoding old QA traces."""
        allowed = self._artifact_path(ref, path)
        if allowed is None:
            return []
        try:
            with allowed.open("rb") as stream:
                stream.seek(0, 2)
                offset = max(0, stream.tell() - 128 * 1024)
                stream.seek(offset)
                lines = stream.read().splitlines()
            if offset:
                lines = lines[1:]
            result = []
            for line in lines:
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        result.append(value)
                except (ValueError, UnicodeError):
                    continue
            return result
        except OSError:
            return []

    def _component_calls(self, ref: _Run) -> list[dict]:
        """Component function observations are independent of model calls."""
        recorded = {}
        for path in (ref.root / "model-captures/component-calls.jsonl",):
            for row in self._rows(ref, path):
                cid = row.get("component_call_id") or row.get("id")
                if not cid:
                    continue
                recorded[cid] = {**row, "id": cid, "component_call_id": cid,
                                 "component_name": row.get("component_name") or row.get("component"),
                                 "seconds": row.get("seconds", row.get("duration_seconds"))}
        return list(recorded.values())

    def _selected_component_call(self, ref: _Run, call: dict) -> dict:
        result = dict(call)
        notices = []
        for field in ("inputs", "output"):
            pathvalue = call.get(field + "_path")
            if field == "inputs" and not pathvalue:
                pathvalue = call.get("input_path")
            path = self._artifact_path(ref, pathvalue)
            result[field] = self._artifact_json(ref, path) if path else call.get(field)
            result[field + "_hash_matches"] = None
            expected = call.get(field + "_sha256") or (call.get("input_sha256") if field == "inputs" else None)
            if path:
                try:
                    result[field + "_hash_matches"] = hashlib.sha256(path.read_bytes()).hexdigest() == expected if expected else None
                except OSError:
                    pass
            if result[field] is None:
                notices.append(f"{field}: {call.get(field + '_not_recorded_reason') or '此组件函数的实际值未记录或尚未返回'}")
            if result[field + "_hash_matches"] is False:
                notices.append(f"{field}: 文件 SHA-256 与组件捕获记录不一致")
        source = call.get("source") or call.get("callable_source")
        if isinstance(source, dict):
            sources = self._compiled_sources(ref, {"workflow": {"name": call.get("component_name"), **source}}, None)
            result["source"] = sources[0] if sources else source
        result["notices"] = notices
        return result

    def _snapshot(self, ref: _Run) -> tuple[dict, dict]:
        report = self._artifact_json(ref, ref.root / "task-flow.report.json") or {}
        events = self._rows(ref, ref.root / "task-flow.events.jsonl")
        config = self._artifact_json(ref, ref.batch / "config-summary.json") or {}
        status_doc = self._artifact_json(ref, ref.batch / "status.json") or {}
        manifest = self._artifact_json(ref, ref.root / "model-captures/manifest.json") or {}
        manifest_summary = {k: manifest.get(k) for k in ("schema_version", "snapshot_status", "source_kind", "source_skill_hash",
                            "source_redacted", "text_hash_scope", "warnings") if k in manifest}
        start = next((e for e in events if e.get("event") == "workflow_started"), {})
        selection = report.get("selection") or start.get("selection") or {}
        calls, captures, logs, unresolved = self._calls(ref, report, start)
        current_attempts = {row.get("task_id") or row.get("name"): row["attempt_id"]
                            for row in report.get("tasks", []) if row.get("attempt_id")}
        for event in events:
            if event.get("event") != "task_started" or not event.get("task_id") or not event.get("attempt_id"):
                continue
            # A report can already include the completed final attempt while
            # its event log still contains the earlier starts.
            if (event["task_id"] in current_attempts and report.get("created_at")
                    and event.get("timestamp") and event["timestamp"] <= report["created_at"]):
                continue
            current_attempts[event["task_id"]] = event["attempt_id"]
        explicit_attempt_tasks = set(current_attempts)
        for call in calls:
            if call.get("task_id") and call.get("attempt_id") and call["task_id"] not in explicit_attempt_tasks:
                current_attempts[call["task_id"]] = call["attempt_id"]
        # The per-batch capture stream may contain more than one run. Keep only
        # task IDs belonging to this run when a selection/task list is available.
        taskmap: dict[str, dict] = {}
        qmaps: dict[str, dict] = {}

        def task(tid: str) -> dict:
            if tid not in taskmap:
                taskmap[tid] = {"id": tid, "index": len(taskmap) + 1, "status": "pending",
                                "question_count": None, "qa_completed": 0,
                                "skill_hash_before": None, "skill_hash_after": None,
                                "attempt_id": current_attempts.get(tid),
                                "failure": None, "error_type": None, "failure_category": None, "failure_history": [],
                                "routing": {}, "optimization": {}, "summary": {}, "questions": []}
                qmaps[tid] = {}
            return taskmap[tid]

        def question(tid: str, row: dict, state: str | None = None):
            qid = row.get("question_id") or row.get("id")
            if not qid:
                return
            target = qmaps[tid].setdefault(qid, {"id": qid, "index": len(qmaps[tid]) + 1,
                                                "status": "pending", "call_ids": []})
            for key in ("question", "prediction", "metrics", "trace", "component_timings",
                        "gold_answers", "retrieved_document_ids", "selection", "label_type"):
                if key in row:
                    target[key] = row[key]
            if "query" in row:
                target["question"] = row["query"]
            if row.get("example_index"):
                target["index"] = row["example_index"]
            if state:
                target["status"] = state
                if state == "completed":
                    target.pop("failure", None)
                    target.pop("error_type", None)
                    target.pop("failure_category", None)

        for row in selection.get("tasks", []):
            tid = row.get("task_id") or row.get("name")
            if tid:
                t = task(tid)
                t["question_count"] = row.get("qa_count")
                for qid in row.get("qa_ids", []):
                    question(tid, {"id": qid})
        for i, row in enumerate(report.get("tasks", []), 1):
            tid = row.get("task_id") or row.get("name") or f"task-{i:04d}"
            t = task(tid)
            if not self._same_attempt(row, t):
                # Question text and annotation remain available, but historical
                # predictions/routing/metrics cannot become the current state.
                for example in row.get("examples", []):
                    question(tid, {k: example[k] for k in ("id", "question_id", "question", "gold_answers") if k in example})
                continue
            t.update({"status": "completed", "routing": row.get("task_routing", {}),
                      "optimization": row.get("optimization", {}), "summary": row.get("summary", {}),
                      "skill_hash_before": row.get("skill_hash_before"),
                      "skill_hash_after": row.get("skill_hash_after"), "attempt_id": row.get("attempt_id")})
            if t["question_count"] is None:
                t["question_count"] = len(row.get("examples", [])) or None
            for example in row.get("examples", []):
                question(tid, example, "completed")
        for event in events:
            tid = event.get("task_id")
            if not tid:
                continue
            t = task(tid)
            name = event.get("event", "")
            if name in {"task_failed", "task_execution_failed"}:
                t["failure_history"].append(event)
            if not self._same_attempt(event, t):
                continue
            if name == "task_started":
                t.update({"index": event.get("task_index", t["index"]),
                          "question_count": event.get("examples", t["question_count"]),
                          "skill_hash_before": event.get("skill_hash_before", t["skill_hash_before"])})
                if t["status"] != "completed":
                    t["status"] = "running"
            elif name == "task_routing_completed":
                t["routing"] = event
            elif name == "task_routing_started":
                t["routing"] = {**event, **t["routing"]}
            elif name == "task_example_started":
                qid = event.get("question_id")
                prior = qmaps[tid].get(qid, {}).get("status")
                question(tid, event, "running" if prior != "completed" else None)
            elif name == "task_example_executed":
                question(tid, event, "completed")
            elif name == "task_evaluation_completed":
                t["summary"] = event.get("task_summary", t["summary"])
                for example in event.get("examples", []):
                    question(tid, example, "completed")
            elif name.startswith("task_optimization_"):
                if name in {"task_optimization_accepted", "task_optimization_rejected", "task_optimization_failed", "task_optimization_skipped", "task_optimization_disabled"}:
                    t["optimization"] = {**t["optimization"], **event,
                                         "status": event.get("status", name.rsplit("_", 1)[1])}
                elif name == "task_optimization_audit_saved":
                    t["optimization"]["audit"] = event
            elif name == "task_completed":
                t["status"] = "completed"
                t["failure"] = None
                t["error_type"] = None
                t["failure_category"] = None
                t["skill_hash_after"] = event.get("skill_hash_after", t["skill_hash_after"])
            elif name in {"task_failed", "task_execution_failed"}:
                # Committed report rows have already survived recovery. An
                # earlier failure in that attempt cannot uncommit them.
                committed = any((row.get("task_id") or row.get("name")) == tid
                                and row.get("attempt_id") == event.get("attempt_id")
                                for row in report.get("tasks", []))
                if not committed and t["status"] != "completed":
                    t["status"] = "failed"
                    t["failure"] = event
                    t["error_type"] = event.get("error_type")
                    t["failure_category"] = event.get("failure_category")
                    if event.get("question_id"):
                        question(tid, event, "failed")
                        qmaps[tid][event["question_id"]].update(failure=event,
                            error_type=event.get("error_type"), failure_category=event.get("failure_category"))
        # Capture metadata can be ahead of the event file during a live write.
        for call in calls:
            tid = call.get("task_id")
            if tid and (not taskmap or tid in taskmap):
                t = task(tid)
                if self._same_attempt(call, t) and call.get("question_id") and call.get("scope") == "primary_qa":
                    question(tid, {"id": call["question_id"]})
        if taskmap:
            calls = [c for c in calls if not c.get("task_id") or c["task_id"] in taskmap]
        # Older runs saved Policy audit bodies before general model capture
        # existed. Their explicit task/attempt paths prove stage ownership. Do
        # not guess which shared API generate_id produced these bodies.
        for t in taskmap.values():
            audit = t.get("optimization", {}).get("audit") or {}
            attempts = audit.get("attempts") or []
            for attempt in attempts:
                artifacts = attempt.get("artifacts") or {}
                stage = attempt.get("stage") or artifacts.get("stage")
                if stage not in {"selection", "editing"}:
                    continue
                callstage = "policy_selection" if stage == "selection" else "policy_editing"
                if any(c.get("task_id") == t["id"] and c.get("stage") == callstage and c.get("capture_available")
                       and self._same_attempt(c, t) and c.get("record_kind") != "optimizer_audit" for c in calls):
                    continue
                prompt_path = self._artifact_path(ref, artifacts.get("prompt_path"))
                response_path = self._artifact_path(ref, artifacts.get("raw_response_path") or artifacts.get("response_path"))
                if not prompt_path or not prompt_path.is_file():
                    continue
                cid = "audit-" + hashlib.sha256(str(prompt_path).encode()).hexdigest()[:20]
                capture = {**artifacts, "generate_id": cid, "response_path": str(response_path) if response_path else None,
                           "stage": callstage, "scope": "optimization", "task_id": t["id"], "role": "policy"}
                captures[cid] = capture
                calls.append({"id": cid, "role": "policy", "model": (report.get("optimizer") or {}).get("model") or (config.get("policy") or {}).get("model"),
                              "stage": callstage, "scope": "optimization", "task_id": t["id"], "question_id": None,
                              "attempt_id": t.get("attempt_id"), "status": "failed" if attempt.get("error") else (
                                  "completed" if response_path and response_path.is_file() else "unknown"),
                              "seconds": None, "timestamp": None, "capture_available": True,
                              "http_attempt_count": None, "node_id": self._call_node(capture),
                              "record_kind": "optimizer_audit", "generate_id_recorded": False})
        for t in taskmap.values():
            request = self._artifact_json(ref, t["routing"].get("request_path"))
            if isinstance(request, dict):
                for row in request.get("questions", []):
                    question(t["id"], row)
            t["questions"] = sorted(qmaps[t["id"]].values(), key=lambda q: q["index"])
            for q in t["questions"]:
                q["call_ids"] = [c["id"] for c in calls if c.get("task_id") == t["id"]
                                 and c.get("question_id") == q["id"] and self._same_attempt(c, t)]
            t["qa_completed"] = sum(q["status"] == "completed" for q in t["questions"])
            if t["question_count"] is None and t["questions"]:
                t["question_count"] = len(t["questions"])
            t["nodes"] = [self._node_summary(t, node, calls) for node in NODES]
            requested = any(e.get("task_id") == t["id"] and e.get("event") == "task_optimization_requested" for e in events)
            optimization_status = t["optimization"].get("status")
            t["optimization_rounds"] = 1 if requested or optimization_status in {"accepted", "rejected", "failed", "skipped"} else 0
            t["optimization_status"] = optimization_status or "unknown"
        for call in calls:
            owner = taskmap.get(call.get("task_id"))
            call["is_current_attempt"] = self._same_attempt(call, owner) if owner else None
        tasks = sorted(taskmap.values(), key=lambda t: t["index"])
        state = "unknown"
        lifecycle = next((e.get("event") for e in reversed(events)
                          if e.get("event") in {"workflow_started", "workflow_completed", "workflow_failed", "workflow_error"}), None)
        if any(t["status"] == "failed" for t in tasks):
            state = "failed"
        elif lifecycle == "workflow_completed":
            state = "completed"
        elif lifecycle in {"workflow_failed", "workflow_error"}:
            state = "failed"
        elif lifecycle == "workflow_started":
            state = "running"
        elif status_doc.get("status") in {"completed", "failed", "stopped", "cancelled"}:
            state = status_doc["status"]
        elif tasks and all(t["status"] == "completed" for t in tasks) and report.get("completed_tasks") == len(tasks):
            state = "completed"
        elif events or calls or status_doc.get("status") in {"started", "running"}:
            state = "running"
        notices = []
        if not captures:
            notices.append("此运行未记录完整模型调用捕获；prompt、模型回答或阶段归属可能缺失。")
        if any(c.get("role") == "executor" for c in calls) and not any(
                c.get("role") == "executor" and c.get("capture_available") for c in calls):
            notices.append("Executor 仅保留 API 元数据，未记录可读取的 prompt 与模型回答正文。")
        if any(c["stage"] == "unknown" for c in calls):
            notices.append("部分 API 调用缺少阶段上下文，已保留为未知，未按顺序猜测归属。")
        if unresolved:
            notices.append(f"共享批次中 {unresolved} 条调用缺少可靠 Group/Run 身份，未归入本运行。")
        if any(c.get("record_kind") == "optimizer_audit" for c in calls):
            notices.append("Policy prompt 与回答由原始 optimizer audit 读取；未猜测对应 API 调用或 HTTP 次数。")
        if state == "running":
            notices.append("运行快照仍可能追加；请手动刷新。正在执行不代表失败。")
        if manifest.get("warnings"):
            notices.append(f"模型观测器记录了 {len(manifest['warnings'])} 条警告；请查看捕获清单，模型调用成败仍以自身记录为准。")
        if manifest.get("snapshot_status") == "incomplete":
            notices.append("模型观测器的初始源码快照未完整写入。")
        if manifest.get("source_redacted"):
            notices.append("初始源码快照含脱敏内容；展示文件与哈希对应保存后的安全副本。")
        result = {"id": ref.id, "name": report.get("run_id") or config.get("run_id") or ref.root.name,
                  "batch": ref.batch.name, "group_id": report.get("group", {}).get("group_id") or config.get("group_id") or start.get("group_id"),
                  "status": state, "executor_model": (config.get("executor") or {}).get("model") or next((c["model"] for c in calls if c["role"] == "executor"), None),
                  "policy_model": (config.get("policy") or {}).get("model") or (report.get("optimizer") or {}).get("model") or start.get("optimizer_model"),
                  "counts": {"tasks_completed": sum(t["status"] == "completed" for t in tasks),
                             "tasks_target": selection.get("task_count") or config.get("max_tasks") or len(tasks) or None,
                             "qa_completed": sum(t["qa_completed"] for t in tasks),
                             "qa_target": selection.get("qa_count") or config.get("total_primary_qa") or (sum(t["question_count"] or 0 for t in tasks) or None),
                             "optimization_rounds": sum(t["optimization_rounds"] for t in tasks),
                             "revisions_accepted": sum(t["optimization"].get("status") == "accepted" for t in tasks)},
                  "tasks": tasks, "calls": calls, "summary": report.get("overall_summary"), "capture_manifest": manifest_summary,
                  "failures": [t["failure"] for t in tasks if t.get("failure")],
                  "call_counts": {"generate_calls": sum(not c.get("record_kind") for c in calls) if any(not c.get("record_kind") for c in calls) else None,
                                  "audit_captures": sum(c.get("record_kind") == "optimizer_audit" for c in calls),
                                  "http_attempts": sum(c.get("http_attempt_count") or 0 for c in calls) if calls and all(c.get("http_attempt_count") is not None for c in calls) else None,
                                  "http_retries": sum(c.get("http_retry_count") or 0 for c in calls) if calls and all(c.get("http_retry_count") is not None for c in calls) else None,
                                  "recovered_calls": sum(c.get("recovered_after_retry") is True for c in calls) if calls and all(c.get("recovered_after_retry") is not None for c in calls) else None},
                  "selection_scope": report.get("selection_scope") or config.get("selection_scope") or start.get("selection_scope"),
                  "revision_validation_mode": report.get("revision_validation_mode") or config.get("revision_validation_mode") or start.get("revision_validation_mode") or "unknown",
                  "experiment_version": report.get("experiment_version") or config.get("experiment_version") or "unknown",
                  "version": report.get("experiment_version") or config.get("experiment_version") or "unknown",
                  "created_at": report.get("created_at") or config.get("created_at"),
                  "notices": notices}
        return result, {"report": report, "events": events, "captures": captures,
                        "component_calls": self._component_calls(ref),
                        "logs": logs, "config": config, "compiles": self._rows(ref, ref.batch / "compile-captures.jsonl")
                        + self._rows(ref, ref.root / "model-captures/compile-captures.jsonl")}

    def _node_summary(self, task: dict, node: str, calls: list[dict]) -> dict:
        relevant = [c for c in calls if c.get("task_id") == task["id"] and c.get("node_id") == node
                    and self._same_attempt(c, task)]
        state = "unknown"
        if relevant:
            state = "running" if any(c["status"] == "running" for c in relevant) else (
                "failed" if any(c["status"] == "failed" for c in relevant) else "completed")
        elif node == "input" and task["questions"]:
            state = "completed"
        elif node in {"manage", "agentic", "components"} and task["routing"].get("plan"):
            state = "completed"
        elif node == "execution" and task["qa_completed"]:
            state = "completed" if task["qa_completed"] == task["question_count"] else "running"
        elif node == "evaluation" and task["summary"]:
            state = "completed"
        elif node in {"policy_select", "policy_edit"} and task["optimization"].get("status"):
            state = "completed" if task["optimization"]["status"] == "accepted" else task["optimization"]["status"]
        elif node == "inheritance" and task.get("skill_hash_after"):
            state = "completed"
        if node == "validation" and task["optimization"].get("verification"):
            verification = task["optimization"]["verification"]
            passed = verification.get("all_passed", verification.get("passed"))
            if passed is not None:
                state = "completed" if passed is True else "failed" if passed is False else "unknown"
        if node == "execution" and task.get("failure") and task["failure"].get("stage") == "qa_execution":
            state = "failed"
        return {"id": node, "label": NODES[node], "status": state,
                "call_ids": [c["id"] for c in relevant],
                "summary": f"{len(relevant)} 条模型调用" if relevant else "依据本地执行记录"}

    def list_runs(self) -> dict:
        result = []
        for ref in self._discover():
            # Listing should not decode every historical trace or prompt. A
            # report is an atomic snapshot; event merging happens on selection.
            paths = [ref.root / "task-flow.report.json", ref.batch / "config-summary.json", ref.batch / "status.json",
                     ref.root / "task-flow.events.jsonl"]
            signature = []
            for path in paths:
                try:
                    stat = path.stat()
                    signature.append((stat.st_mtime_ns, stat.st_size))
                except OSError:
                    signature.append(None)
            key = tuple(signature)
            cached = self._summaries.get(ref.id)
            if cached and cached[0] == key:
                result.append(cached[1])
                continue
            report = self._artifact_json(ref, paths[0]) or {}
            config = self._artifact_json(ref, paths[1]) or {}
            status = self._artifact_json(ref, paths[2]) or {}
            tasks = report.get("tasks", [])
            selection = report.get("selection", {})
            task_target = selection.get("task_count") or config.get("max_tasks")
            completed = report.get("completed_tasks")
            if completed is None:
                completed = len(status.get("completed_tasks", [])) if status.get("completed_tasks") is not None else (len(tasks) if tasks else None)
            state = status.get("status", "unknown")
            if state == "started":
                state = "running"
            if task_target and completed == task_target:
                state = "completed"
            elif not status and completed:
                state = "unknown"
            for event in reversed(self._tail_rows(ref, paths[3])):
                name = event.get("event")
                if name in {"task_execution_failed", "task_failed"}:
                    if any((t.get("task_id") or t.get("name")) == event.get("task_id")
                           and t.get("attempt_id") == event.get("attempt_id") for t in tasks):
                        continue
                    state = "failed"
                    break
                if name in {"workflow_failed", "workflow_error"}:
                    state = "failed"
                    break
                if name == "workflow_completed":
                    state = "completed"
                    break
                if name in {"workflow_started", "task_started"}:
                    if not (task_target and completed == task_target):
                        state = "running"
                    break
            captures_exist = any(p.is_file() for p in [ref.batch / "capture-log.jsonl", ref.root / "model-captures/capture-log.jsonl"])
            summary = {"id": ref.id, "name": report.get("run_id") or config.get("run_id") or ref.root.name,
                       "batch": ref.batch.name, "group_id": report.get("group", {}).get("group_id") or config.get("group_id"),
                       "status": state, "executor_model": (config.get("executor") or {}).get("model"),
                       "policy_model": (config.get("policy") or {}).get("model") or (report.get("optimizer") or {}).get("model"),
                       "created_at": report.get("created_at") or config.get("created_at"),
                       "revision_validation_mode": report.get("revision_validation_mode") or config.get("revision_validation_mode") or "unknown",
                       "experiment_version": report.get("experiment_version") or config.get("experiment_version") or "unknown",
                       "version": report.get("experiment_version") or config.get("experiment_version") or "unknown",
                       "counts": {"tasks_completed": completed, "tasks_target": task_target,
                                  "qa_completed": sum(len(t.get("examples", [])) for t in tasks) if tasks else None,
                                  "qa_target": selection.get("qa_count") or config.get("total_primary_qa"),
                                  "optimization_rounds": sum(t.get("optimization", {}).get("status") in {"accepted", "rejected", "failed", "skipped"} for t in tasks) if tasks else None,
                                  "revisions_accepted": sum(t.get("optimization", {}).get("status") == "accepted" for t in tasks) if tasks else None},
                       "notices": [] if captures_exist else ["此运行未记录完整模型调用捕获。"]}
            self._summaries[ref.id] = (key, summary)
            result.append(summary)
        result.sort(key=lambda r: (r.get("created_at") or "", not bool(r.get("notices"))), reverse=True)
        return redact({"runs": result})

    def get_run(self, run_id: str) -> dict:
        return redact(self._snapshot(self._find(run_id))[0])

    def _selected_call(self, ref: _Run, call: dict, raw: dict) -> dict:
        cid = call["id"]
        cap = raw["captures"].get(cid, {})
        result = dict(call)
        notices = []
        for label, pathkey in (("prompt", "prompt_path"), ("system_prompt", "system_prompt_path"),
                               ("response", "response_path")):
            content, error = self._read_text(ref, cap.get(pathkey)) if cap.get(pathkey) else (None, "未记录")
            result[label] = content
            result[f"{label}_sha256"] = cap.get(f"{label}_sha256")
            expected = cap.get(f"{label}_sha256")
            result[f"{label}_hash_matches"] = None
            if content is not None and expected:
                path = self._artifact_path(ref, cap[pathkey])
                try:
                    result[f"{label}_hash_matches"] = hashlib.sha256(path.read_bytes()).hexdigest() == expected
                    if result[f"{label}_hash_matches"] is False:
                        notices.append(f"{label}: 文件 SHA-256 与捕获记录不一致")
                except OSError:
                    pass
            if error:
                notices.append(f"{label}: {cap.get(label + '_not_recorded_reason') or cap.get('text_not_recorded_reason') or error}")
            result[f"{label}_redacted"] = cap.get(f"{label}_redacted")
        result["text_hash_scope"] = cap.get("text_hash_scope", "unknown")
        result["http_attempts"] = raw["logs"].get(cid, [])
        result["notices"] = notices
        return result

    def _source(self, ref: _Run, path: Path, label: str, expected: str | None = None) -> dict | None:
        path = path.resolve()
        allowed = self._artifact_path(ref, path)
        if allowed is None:
            # Repository fallbacks are fixed source code/Skill paths chosen by
            # this module; settings and arbitrary repository files stay denied.
            if not (_within(path, self.repo_root / "framework") or _within(path, self.repo_root / "self_optimization")):
                return None
            if path.suffix not in {".py", ".md", ".yaml"} or _blocked(path):
                return None
        try:
            data = path.read_bytes()
            if len(data) > 2 * 1024 * 1024:
                return None
            digest = hashlib.sha256(data).hexdigest()
            return {"label": label, "path": str(path), "text": data.decode("utf-8-sig"),
                    "sha256": digest, "expected_sha256": expected,
                    "hash_matches": digest == expected if expected else None,
                    "source_kind": "captured_execution" if expected and digest == expected else (
                        "frozen_snapshot" if "source-code" in path.parts or "source-skills" in path.parts else "current_reference"),
                    "provenance": "captured_execution" if expected and digest == expected else (
                        "frozen_source" if _within(path, ref.batch / "source-code") or _within(path, ref.batch / "source-skills") else "current_source")}
        except (OSError, UnicodeError):
            return None

    def _compiled_sources(self, ref: _Run, compilation: dict, component: str | None) -> list[dict]:
        entries = [compilation.get("workflow") or {}] + (compilation.get("components") or [])
        results = []
        for entry in entries:
            if not entry.get("path"):
                continue
            if component and component not in {entry.get("name"), entry.get("slot")}:
                continue
            expected = entry.get("file_sha256")
            snapshot = self._artifact_path(ref, entry.get("snapshot_path"))
            if snapshot:
                source = self._source(ref, snapshot, entry.get("name", "执行源码"), entry.get("snapshot_sha256") or expected)
                if source and source["hash_matches"]:
                    source["original_path"] = str(_path(entry["path"]))
                    source["original_sha256"] = expected
                    source["redacted"] = entry.get("redacted", False)
                    results.append(source)
                    continue
            path = self._artifact_path(ref, entry["path"])
            if path is None:
                continue
            options = [path]
            try:
                relative = path.relative_to(ref.root / "skills")
                options += [ref.batch / "source-skills" / relative, ref.root / "model-captures/source-skills" / relative]
                # Revisions preserve the edited Skill's previous file. Resolve
                # by hash rather than presenting today's code as historical.
                for rev in _children(ref.root / "revisions"):
                    proposal = _json(rev / "proposal.json", {})
                    selected = proposal.get("selected_skill")
                    if selected == entry.get("name") or (len(relative.parts) > 1 and selected == relative.parts[1]):
                        options.append(rev / "before" / Path(*relative.parts[2:]))
            except ValueError:
                pass
            candidates = [s for p in options if (s := self._source(ref, p, entry.get("name", "执行源码"), expected))]
            source = next((s for s in candidates if s["hash_matches"]), candidates[0] if candidates else None)
            if source:
                results.append(source)
        return results

    def _logic_sources(self, ref: _Run, node: str, task: dict, raw: dict,
                       question_id: str | None, component: str | None) -> list[dict]:
        fixed = {"input": ["framework/data_groups.py"],
                 "manage": ["framework/selection.py"], "agentic": ["framework/selection.py"],
                 "components": ["framework/compiler.py", "framework/selection.py"],
                 "execution": ["framework/compiler.py"],
                 "evaluation": ["framework/evaluation/evaluator.py", "framework/evaluation/generation.py"],
                 "policy_select": ["self_optimization/optimizer.py"],
                 "policy_edit": ["self_optimization/optimizer.py", "self_optimization/contracts.py"],
                 "validation": ["self_optimization/execution_validation.py"],
                 "inheritance": ["self_optimization/task_flow.py", "self_optimization/checkpoint.py"]}
        sources = []
        for relative in fixed[node]:
            frozen = ref.root / "model-captures/source-code" / relative
            if not frozen.is_file():
                frozen = ref.batch / "source-code" / relative
            chosen = frozen if frozen.is_file() else self.repo_root / relative
            source = self._source(ref, chosen, relative)
            if source:
                sources.append(source)
        compiles = [c for c in raw["compiles"] if c.get("task_id") == task["id"]
                    and self._same_attempt(c, task)
                    and (not question_id or c.get("question_id") == question_id)
                    and ((c.get("scope") == "revision_validation") == (node == "validation"))]
        if node in {"execution", "components", "agentic", "validation"} and compiles:
            sources.extend(self._compiled_sources(ref, compiles[0], component))
        plan = task.get("routing", {}).get("plan", {})
        skills = []
        if node == "manage":
            skills = [("manage", plan.get("manage_skill"))]
        elif node == "agentic":
            skills = [("agentic", plan.get("agentic_skill"))]
        elif node in {"components", "execution"}:
            skills = [("components", name) for slot, names in plan.get("component_bindings", {}).items()
                      for name in names if not component or component in {slot, name}]
        for kind, name in skills:
            if not name or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                continue
            skill_path = ref.root / "model-captures/source-skills" / kind / name / "SKILL.md"
            if not skill_path.is_file():
                skill_path = ref.batch / "source-skills" / kind / name / "SKILL.md"
            source = self._source(ref, skill_path, f"初始 Skill · {name}")
            if source:
                sources.append(source)
        return sources

    def get_node(self, run_id: str, task_id: str, node_id: str,
                 question_id: str | None = None, call_id: str | None = None,
                 component: str | None = None) -> dict:
        if node_id not in NODES:
            raise ValueError("未知框架节点")
        ref = self._find(run_id)
        run, raw = self._snapshot(ref)
        task = next((t for t in run["tasks"] if t["id"] == task_id), None)
        if task is None:
            raise KeyError("任务不存在")
        selected_question = next((q for q in task["questions"] if q["id"] == question_id), None)
        if question_id and selected_question is None:
            raise KeyError("问题不存在")
        call_node = ("validation" if node_id == "validation" else "execution") if component else node_id
        qa_node_calls = [c for c in run["calls"] if c.get("task_id") == task_id and c.get("node_id") == call_node
                         and (not question_id or c.get("question_id") in {question_id, None})]
        observations = [c for c in raw["component_calls"] if node_id in {"execution", "components", "validation"} and c.get("task_id") == task_id
                        and (not question_id or c.get("question_id") == question_id)
                        and ((c.get("scope") == "revision_validation") == (node_id == "validation"))
                        and (not component or component in {c.get("component_name"), c.get("slot")})]
        observation_ids = {c["component_call_id"] for c in observations}
        all_node_calls = [c for c in qa_node_calls if not component or component in {c.get("component"), c.get("slot")}
                          or c.get("component_call_id") in observation_ids]
        unattributed_qa_calls = [c for c in qa_node_calls if component and not c.get("component") and not c.get("slot")
                                and not c.get("component_call_id") and self._same_attempt(c, task)]
        calls = [c for c in all_node_calls if self._same_attempt(c, task)]
        historical_calls = [c for c in all_node_calls if not self._same_attempt(c, task)]
        chosen = next((c for c in all_node_calls if c["id"] == call_id), None) if call_id else (calls[0] if calls else None)
        chosen_unattributed = False
        if call_id and chosen is None:
            chosen = next((c for c in unattributed_qa_calls if c["id"] == call_id), None)
            chosen_unattributed = chosen is not None
        if call_id and chosen is None:
            raise KeyError("该节点没有此模型调用")
        viewing_history = chosen is not None and not self._same_attempt(chosen, task)
        if viewing_history:
            calls = [chosen]
        plan = task.get("routing", {}).get("plan", {})
        opt = task.get("optimization", {})
        notices = []
        component_calls = [self._selected_component_call(ref, c) for c in observations if self._same_attempt(c, task)]
        selected_component_call = next((c for c in component_calls if chosen and c["id"] == chosen.get("component_call_id")),
                                       component_calls[0] if component_calls else None)
        if component and unattributed_qa_calls:
            notices.append("日志仅关联到该QA，未逐组件标注，不能视为组件专属；请返回整个 QA 手动检视模型调用。")
        if chosen_unattributed:
            notices.append("当前显式查看的模型调用只有 QA 级关联，未证实属于所选组件。")
        component_status = None
        if component and component_calls:
            states = {c.get("status") for c in component_calls}
            component_status = "running" if "running" in states else "failed" if "failed" in states else "completed" if states == {"completed"} else "unknown"
        descriptions = {
            "input": "同一 Task 的问题集合与语料元数据。Task 共享路由时整组问题参与一次选择。",
            "manage": "Manage 根据任务请求生成统一指导；下一阶段依据该指导选择 Agentic Skill。",
            "agentic": "在 Manage 指导下选择可执行 Agentic 工作流。",
            "components": "为工作流插槽绑定 Component Skills；本地检索器和重排器不必调用语言模型。",
            "execution": "执行选定工作流，显示每题检索、生成、Critic反馈与停止条件。",
            "evaluation": "对实际输出计算检索和生成指标。",
            "policy_select": "Policy 根据本 Task 的结果选择需要改进的 Skill。",
            "policy_edit": "Policy 提交受约束修改，保留提案、差异与实际文件证据。",
            "validation": "执行修改后的 Skill，检查约定的运行与接口条件；性能门槛以记录为准。",
            "inheritance": "核对修改前后 Skill 集合哈希与下一 Task 的起始版本。",
        }
        inputs: Any = {}
        output: Any = {}
        if node_id == "input":
            inputs = {"group": raw["report"].get("group"), "selection_scope": run["selection_scope"],
                      "questions": [{k: q.get(k) for k in ("id", "index", "question")} for q in task["questions"]]}
            output = {"question_count": task["question_count"], "routing_request": self._artifact_json(ref, task["routing"].get("request_path"))}
        elif node_id in {"manage", "agentic", "components"}:
            inputs = {"routing": task["routing"], "skill_hash": task["skill_hash_before"]}
            keys = {"manage": ["manage_skill", "manage_guidance", "manage_reason"],
                    "agentic": ["agentic_skill", "agentic_reason"],
                    "components": ["component_bindings", "component_reason"]}[node_id]
            output = {k: plan.get(k) for k in keys}
        elif node_id == "execution":
            inputs = {"question": selected_question.get("question") if selected_question else None,
                      "plan": plan, "skill_hash": task["skill_hash_before"]}
            output = selected_question or {"questions": task["questions"]}
        elif node_id == "evaluation":
            inputs = {"question": selected_question, "qa_completed": task["qa_completed"]}
            output = selected_question.get("metrics", {}) if selected_question else task["summary"]
        elif node_id in {"policy_select", "policy_edit"}:
            inputs = {"task_summary": task["summary"], "skill_hash": task["skill_hash_before"]}
            output = {"selected_skill": opt.get("selected_skill"), "rationale": opt.get("rationale"),
                      "status": opt.get("status"), "revision": opt.get("revision"), "audit": opt.get("audit")}
        elif node_id == "validation":
            inputs = {"revision": opt.get("revision"), "plan": plan}
            output = opt.get("verification", {})
        elif node_id == "inheritance":
            following = next((t for t in run["tasks"] if t["index"] == task["index"] + 1), None)
            output = {"skill_hash_before": task["skill_hash_before"], "skill_hash_after": task["skill_hash_after"],
                      "next_task_id": following["id"] if following else None,
                      "next_skill_hash_before": following["skill_hash_before"] if following else None,
                      "inherited": (task["skill_hash_after"] == following["skill_hash_before"])
                      if following and task["skill_hash_after"] and following["skill_hash_before"] else None}
            inputs = {"revision": opt.get("revision"), "optimization_status": opt.get("status")}
        revision = opt.get("revision") or {}
        versions: Any = {"before": task["skill_hash_before"], "after": task["skill_hash_after"], "revision": revision}
        if node_id in {"policy_edit", "inheritance", "validation"}:
            diff, error = self._read_text(ref, revision.get("diff_path")) if revision.get("diff_path") else (None, None)
            versions["diff"] = diff
            versions["proposal"] = self._artifact_json(ref, str(_path(revision["revision_dir"]) / "proposal.json")) if revision.get("revision_dir") else None
            if error:
                notices.append(error)
        sources = self._logic_sources(ref, node_id, task, raw, question_id, component)
        if component:
            if selected_component_call:
                inputs = selected_component_call.get("inputs")
                output = selected_component_call.get("output")
                notices.extend(selected_component_call.get("notices", []))
                actual_source = selected_component_call.get("source")
                if isinstance(actual_source, dict) and actual_source.get("text"):
                    sources = [actual_source] + [s for s in sources if s.get("sha256") != actual_source.get("sha256")]
            else:
                inputs = {"question": selected_question.get("question") if selected_question else None,
                          "component": component, "component_function_inputs_recorded": False}
                output = {"component_function_output_recorded": False,
                          "recorded_component_timings": [row for row in (selected_question or {}).get("component_timings", [])
                                                         if component in {row.get("component"), row.get("slot")} ]}
                notices.append("此运行未记录该组件逐次函数输入与输出；未将 QA 总体预测或其他组件的模型回答代作该组件输出。")
        if chosen_unattributed:
            inputs = {k: chosen.get(k) for k in ("task_id", "attempt_id", "question_id", "stage", "scope", "skill_hash")}
            inputs["association"] = "question_only"
            output = {"record_kind": "unattributed_qa_model_call", "call_id": chosen["id"]}
        if viewing_history:
            # An explicit historical call can be inspected without making its
            # prompt/response look like the current Task's route or prediction.
            inputs = {k: chosen.get(k) for k in ("task_id", "attempt_id", "question_id", "stage", "scope", "skill_hash")}
            inputs["recorded_context_only"] = True
            output = {"record_kind": "historical_model_call", "call_id": chosen["id"]}
            sources = []
            versions = {"before": chosen.get("skill_hash"), "after": None, "revision": None}
            notices.append("当前查看历史 attempt 的独立模型调用；未与当前 Task 的规划、预测或源码混合。")
        elif historical_calls:
            notices.append(f"此节点另有 {len(historical_calls)} 条历史或 attempt 身份未知的调用，未混入当前 Task；可从调用记录单独查看。")
        if any(s.get("hash_matches") is False for s in sources):
            notices.append("部分源码与执行捕获哈希不一致；已标记当前源码，不能据此认定历史实现。")
        if any(s.get("source_kind") == "current_reference" for s in sources):
            notices.append("当前参考源码仅用于解释框架，未记录为本次运行快照，不能据此认定历史实现。")
        if not calls and node_id in {"manage", "agentic", "components", "policy_select", "policy_edit", "execution"}:
            notices.append("此节点没有可归属的完整模型调用记录；未补造 prompt 或回答。")
        relevant_events = [e for e in raw["events"] if e.get("task_id") == task_id
                           and e.get("attempt_id") == (chosen.get("attempt_id") if viewing_history else task.get("attempt_id"))
                           and (not question_id or e.get("question_id") in {question_id, None})]
        result = {"title": NODES[node_id], "id": node_id, "task_id": task_id,
                  "question_id": question_id, "status": chosen["status"] if viewing_history else component_status or next(n["status"] for n in task["nodes"] if n["id"] == node_id),
                  "model": chosen.get("model") if chosen else (run["policy_model"] if node_id.startswith("policy") else None),
                  "description": descriptions[node_id], "inputs": inputs, "output": output,
                  "calls": calls, "selected_call": self._selected_call(ref, chosen, raw) if chosen else None,
                  "component_calls": component_calls, "selected_component_call": selected_component_call,
                  "unattributed_qa_calls": unattributed_qa_calls,
                  "model_call_association": "question_only" if chosen_unattributed else "component" if component and calls else "question" if not component else "unrecorded",
                  "logic_sources": sources, "verification": None if viewing_history else opt.get("verification"),
                  "failure": None if viewing_history else task.get("failure"),
                  "skill_versions": versions, "events": relevant_events,
                  "evidence": {"run_path": str(ref.root), "capture_count": len(calls), "source": "recorded_artifacts",
                               "current_attempt_id": task.get("attempt_id"), "historical_call_count": len(historical_calls),
                               "viewing_historical_call": viewing_history, "capture_manifest": run.get("capture_manifest"),
                               "component_call_count": len(component_calls), "unattributed_qa_call_count": len(unattributed_qa_calls)},
                  "evaluation_only_annotations": [{"question_id": q["id"], "gold_answers": q.get("gold_answers")}
                                                  for q in task["questions"] if q.get("gold_answers") is not None] if node_id == "input" else None,
                  "notices": notices}
        return redact(result)
