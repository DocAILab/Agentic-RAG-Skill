"""Best-effort, credential-redacted model I/O observations; never a model transport.

Only allowlisted source files and metadata are inspected. Model configurations,
environment variables, headers, endpoints and credential attributes are not read.
Persisted SHA256 values identify the saved (possibly redacted) UTF-8 bytes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
import time
import uuid
import warnings


STATISTICS_KEYS = (
    "requests", "succeeded", "failed", "attempts", "retries",
    "response_structure_errors", "provider_errors", "http_errors",
    "transport_errors", "response_content_errors", "other_api_errors",
)
CONTEXT_KEYS = {"group_id", "task_id", "attempt_id", "question_id", "scope", "edit_mode"}
COMPONENT_IO_LIMITS = {"max_bytes": 262144, "max_collection_items": 50,
                       "max_string_bytes": 32768, "max_depth": 8, "max_nodes": 1000}
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9._-]+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[^\s\"'<>;,]+", re.IGNORECASE)
_FIELD = re.compile(
    r'''(?ix)(["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|password|secret|credentials)["']?\s*[:=]\s*)'''
    r'''("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}\]\n]+)'''
)


class ModelCaptureWarning(UserWarning):
    """Observation failed; the underlying model outcome is unchanged."""


def redact_text(value: str) -> str:
    value = _FIELD.sub(lambda match: match.group(1) + '"[REDACTED]"', value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    return _SK_KEY.sub("[REDACTED]", value)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _private(path):
    name = path.name.lower()
    return ("settings" in name or "credential" in name or "secret" in name
            or name == ".env" or name.startswith(".env."))


def _model_label(client, key):
    try:
        value = getattr(client, key, None)
        return redact_text(value) if isinstance(value, str) else None
    except Exception:
        return None


def _stage(role, system):
    if not isinstance(system, str):
        return "unknown" if role == "policy" else "generation"
    if role == "policy":
        if system.startswith("Select exactly one participating Skill to improve"):
            return "policy_selection"
        if system.startswith("You are editing one already-selected Skill"):
            return "policy_editing"
        return "unknown"
    for marker, stage in (("at the Manage stage", "manage"),
                          ("at the Agentic selection stage", "agentic"),
                          ("at the Component selection stage", "component")):
        if marker in system:
            return stage
    return "generation"


class ModelCaptureRecorder:
    """One recorder per run, with Task baseline and worker-local QA context.

    Context deliberately does not infer Component ownership from prompt text.
    No calls to request_statistics() or transport interception are performed.
    """

    def __init__(self, run_root, *, source_skill_root, skill_hash, source_hash=None,
                 warning_callback=None, corpus_fingerprint=None):
        self.root = Path(run_root) / "model-captures"
        self.skill_hash = skill_hash
        self.lock = RLock()
        self.base_context = {}
        self.local_context = ContextVar("model_capture_context_" + uuid.uuid4().hex, default={})
        self.component_context = ContextVar("component_capture_context_" + uuid.uuid4().hex, default={})
        self.corpus_fingerprint = corpus_fingerprint
        self.warning_callback = warning_callback
        self._manifest_can_update = True
        self.manifest = {"schema_version": 1, "created_at": _utc(), "source_kind": "frozen_snapshot",
                         "source_code_scope": "recorder_initialization", "source_skill_hash": source_hash,
                         "snapshot_status": "incomplete",
                         "capture_log": str(self.root / "capture-log.jsonl"),
                         "compile_log": str(self.root / "compile-captures.jsonl"),
                         "component_log": str(self.root / "component-calls.jsonl"),
                         "vector_index_log": str(self.root / "vector-index-events.jsonl"),
                         "component_io_limits": dict(COMPONENT_IO_LIMITS),
                         "text_hash_scope": "persisted_redacted_utf8", "warnings": [],
                         "http_attempts": "unknown unless client exposes last_request_statistics"}
        self._observe("initialize", lambda: self._initialize(Path(source_skill_root)))

    def _write(self, path, content, *, exclusive=False):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
        with os.fdopen(os.open(path, flags, 0o600), "wb") as handle:
            handle.write(content)

    def _json(self, path, value):
        content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
        try:
            self._write(temporary, content, exclusive=True)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _append(self, filename, row):
        content = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self.root / filename
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "ab") as handle:
                handle.write(content)

    def _observe(self, operation, function):
        try:
            return function()
        except Exception as exc:
            self._warning(operation, exc)
            return None

    def _warning(self, operation, exc):
        # Error messages can contain credentials or model payloads. Only the
        # observer operation and exception type are persisted here.
        warning = {"timestamp": _utc(), "operation": operation, "error_type": type(exc).__name__}
        with self.lock:
            self.manifest["warnings"].append(warning)
            try:
                if self._manifest_can_update:
                    self._json(self.root / "manifest.json", self.manifest)
            except Exception:
                pass
            try:
                self._append("observer-warnings.jsonl", warning)
            except Exception:
                pass
        if self.warning_callback is not None:
            try:
                self.warning_callback(warning)
            except Exception:
                pass
        try:
            warnings.warn(f"Model I/O capture {operation} failed ({type(exc).__name__}); see observer manifest/event.",
                          ModelCaptureWarning, stacklevel=3)
        except Exception:
            pass  # A warnings-as-errors policy must not retry a successful API.

    def _snapshot_file(self, source, target):
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        safe = redact_text(text).encode("utf-8")
        if target.exists():
            if target.read_bytes() != safe:
                raise FileExistsError("Frozen source conflicts with the existing snapshot")
        else:
            self._write(target, safe, exclusive=True)
        return {"path": str(target), "source_sha256": _hash(raw), "sha256": _hash(safe),
                "redacted": raw != safe}

    def _initialize(self, source_skills):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            self._manifest_can_update = False
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or not isinstance(saved.get("warnings"), list):
                raise ValueError("Existing capture manifest is invalid")
            expected_hash = self.manifest["source_skill_hash"]
            self.manifest = saved
            self._manifest_can_update = True
            if saved.get("source_skill_hash") != expected_hash:
                raise ValueError("Existing capture baseline has a different source identity")
            # Resume appends; complete snapshots stay frozen. Interrupted
            # snapshots may safely reuse only byte-identical existing files.
            if saved.get("snapshot_status") == "complete":
                return
        files = []
        repo = Path(__file__).resolve().parents[1]
        for package in ("framework", "self_optimization"):
            for source in sorted((repo / package).rglob("*.py")):
                relative = source.relative_to(repo / package)
                if ("skills" not in relative.parts and "__pycache__" not in relative.parts
                        and not _private(source) and not source.is_symlink()):
                    target = self.root / "source-code" / package / relative
                    files.append(self._snapshot_file(source, target))
        for source in sorted(source_skills.rglob("*")):
            if not source.is_file() or source.is_symlink() or _private(source):
                continue
            relative = source.relative_to(source_skills)
            if "__pycache__" in relative.parts or not (
                source.name in {"SKILL.md", "ragskill.yaml"}
                or ("scripts" in relative.parts and source.suffix == ".py")
            ):
                continue
            files.append(self._snapshot_file(source, self.root / "source-skills" / relative))
        self.manifest.update(source_code_root=str(self.root / "source-code"),
                             source_skills_root=str(self.root / "source-skills"), source_files=files,
                             snapshot_status="complete", source_redacted=any(item["redacted"] for item in files))
        self._json(manifest_path, self.manifest)

    def wrap(self, client, role):
        return CapturedModel(client, role, self)

    def update_context(self, **fields):
        with self.lock:
            self.base_context.update({key: redact_text(value) if isinstance(value, str) else value
                                      for key, value in fields.items() if key in CONTEXT_KEYS})

    @contextmanager
    def context(self, **fields):
        """Override only the calling execution context; never mutate Task state.

        Submit a fresh copy_context().run when spawning additional nested Python
        threads that must inherit this scope. Independent QA workers set their
        own scope, so overlapping tasks cannot overwrite another QA's identity.
        """
        fields = {key: redact_text(value) if isinstance(value, str) else value
                  for key, value in fields.items() if key in CONTEXT_KEYS}
        token = self.local_context.set({**self.local_context.get(), **fields})
        try:
            yield
        finally:
            self.local_context.reset(token)

    def _context(self):
        with self.lock:
            return {**self.base_context, **self.local_context.get(), **self.component_context.get()}

    def record_vector_index(self, record):
        """Record one real index operation without changing its outcome."""
        def write():
            allowed = {
                "operation", "status", "cache_key", "cache_source", "cache_hit",
                "document_count", "dimension", "cache_path", "duration_seconds",
                "build_seconds", "load_seconds", "disk_lookup_seconds", "wait_seconds",
                "query_seconds", "query_embedding_seconds", "search_seconds",
                "error_type", "error", "component", "slot",
            }
            row = {key: value for key, value in record.items() if key in allowed}
            # Context ownership is supplied by the recorder, never a component.
            row.update(self._context(), timestamp=_utc(), event_id=uuid.uuid4().hex)
            content, _, _ = self._component_payload(row)
            self._append("vector-index-events.jsonl", json.loads(content))
        self._observe("record_vector_index", write)

    def _text(self, row, field, value, directory):
        row[field + "_path"] = None
        row[field + "_sha256"] = None
        row[field + "_not_recorded_reason"] = "observer_io_error"
        if not isinstance(value, str):
            row[field + "_not_recorded_reason"] = "non_text_or_unavailable"
            return
        safe = redact_text(value)
        content = safe.encode("utf-8")
        path = directory / (field + ".txt")
        self._write(path, content, exclusive=True)
        row[field + "_path"] = str(path)
        row[field + "_sha256"] = _hash(content)
        row[field + "_redacted"] = safe != value
        row[field + "_characters"] = len(safe)
        row.pop(field + "_not_recorded_reason", None)

    def start(self, client, role, args, kwargs):
        prompt = args[0] if args else kwargs.get("prompt")
        system = kwargs.get("system")
        parameters = {}
        if "temperature" in kwargs:
            value = kwargs["temperature"]
            if type(value) in {int, float} and math.isfinite(value):
                parameters["temperature"] = value
        if "max_tokens" in kwargs and (kwargs["max_tokens"] is None or type(kwargs["max_tokens"]) is int):
            parameters["max_tokens"] = kwargs["max_tokens"]
        row = {"schema_version": 1, "timestamp": _utc(), "generate_id": uuid.uuid4().hex,
               "source_kind": "captured_execution", "role": role, "model": _model_label(client, "model"),
               "provider": _model_label(client, "provider"), "stage": _stage(role, system),
               "system_prompt_supplied": isinstance(system, str),
               "parameters": parameters, "parameters_scope": "explicit_generate_arguments_only",
               "scope": "unknown", "task_id": None, "attempt_id": None, "question_id": None,
               "component_call_id": None, "component_name": None, "slot": None,
               "skill_hash": None, "text_hash_scope": "persisted_redacted_utf8",
               "request_statistics": None, "request_statistics_availability": "unknown", **self._context()}
        directory = self.root / "captures" / role / row["generate_id"]
        self._observe("skill_hash", lambda: row.update(skill_hash=self.skill_hash()))
        for field, value in (("prompt", prompt), ("system_prompt", system or "")):
            self._observe("write_" + field, lambda f=field, v=value: self._text(row, f, v, directory))
        row.update(status="running", seconds=None, error_type=None, response_path=None,
                   response_sha256=None, response_not_recorded_reason="generation_in_progress")
        self._observe("append_running", lambda: self._append("capture-log.jsonl", row))
        return row, directory

    def finish(self, client, observation, result, error, seconds):
        row, directory = observation
        row.update(status="failed" if error is not None else "completed", seconds=seconds,
                   finished_at=_utc(),
                   error_type=type(error).__name__ if error is not None else None)
        self._observe("write_response", lambda: self._text(row, "response", result, directory))
        def statistics():
            values = getattr(client, "last_request_statistics", None)
            values = values() if callable(values) else values
            if isinstance(values, Mapping):
                saved = {key: values[key] for key in STATISTICS_KEYS
                         if key in values and type(values[key]) is int and values[key] >= 0}
                if saved:
                    row.update(request_statistics=saved, request_statistics_availability="client_reported_last_call")
        self._observe("last_request_statistics", statistics)
        self._observe("append_capture", lambda: self._append("capture-log.jsonl", row))

    def _callable_source(self, function, name):
        source = Path(function.__code__.co_filename).resolve()
        if _private(source) or source.suffix != ".py":
            raise ValueError("Callable source is not an allowlisted Python file")
        raw = source.read_bytes()
        safe = redact_text(raw.decode("utf-8")).encode("utf-8")
        target = self.root / "executed-sources" / (_hash(safe) + ".py")
        with self.lock:
            if not target.exists():
                self._write(target, safe, exclusive=True)
            elif target.read_bytes() != safe:
                raise ValueError("Executed source snapshot hash/content mismatch")
        return {"name": name, "path": str(source), "file_sha256": _hash(raw),
                "snapshot_path": str(target), "snapshot_sha256": _hash(safe),
                "source_kind": "captured_execution", "redacted": safe != raw}

    def _component_payload(self, value):
        """Bound observations before serialization, without copying a full Corpus."""
        omissions, redacted, nodes = [], False, 0
        def summary(item, path, reason):
            note = {"path": path, "reason": reason}
            result = {"recording": "summary", "type": type(item).__name__, "reason": reason}
            if isinstance(item, (Mapping, Sequence)):
                result["count"] = len(item)
                note["count"] = len(item)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                result["sample_ids"] = [redact_text(entry["id"]) for entry in item[:5]
                                        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)
                                        and len(entry["id"]) <= 256]
                result["sample_id_policy"] = "first_five_records_with_ids_up_to_256_characters"
                if self.corpus_fingerprint is not None and len(item) > COMPONENT_IO_LIMITS["max_collection_items"]:
                    result["corpus_fingerprint"] = self.corpus_fingerprint
            if len(omissions) < 100:
                omissions.append(note)
            return result
        def convert(item, path="$", depth=0):
            nonlocal nodes, redacted
            nodes += 1
            if nodes > COMPONENT_IO_LIMITS["max_nodes"] or depth > COMPONENT_IO_LIMITS["max_depth"]:
                return summary(item, path, "structure_limit")
            if item is None or type(item) in {bool, int}:
                return item
            if type(item) is float and math.isfinite(item):
                return item
            if isinstance(item, str):
                if len(item.encode("utf-8")) > COMPONENT_IO_LIMITS["max_string_bytes"]:
                    return summary(item, path, "text_exceeds_byte_limit")
                safe = redact_text(item)
                redacted = redacted or safe != item
                return safe
            if isinstance(item, Mapping):
                if len(item) > COMPONENT_IO_LIMITS["max_collection_items"]:
                    return summary(item, path, "collection_exceeds_item_limit")
                result = {}
                for key, entry in item.items():
                    if not isinstance(key, str):
                        return summary(item, path, "non_text_mapping_key")
                    safe_key = redact_text(key)
                    redacted = redacted or safe_key != key
                    if re.fullmatch(r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|password|secret|credentials)", key):
                        result[safe_key] = "[REDACTED]"
                        redacted = True
                    else:
                        result[safe_key] = convert(entry, path + "." + safe_key, depth + 1)
                return result
            if isinstance(item, Sequence) and not isinstance(item, bytes):
                if len(item) > COMPONENT_IO_LIMITS["max_collection_items"]:
                    return summary(item, path, "collection_exceeds_item_limit")
                return [convert(entry, f"{path}[{index}]", depth + 1) for index, entry in enumerate(item)]
            return summary(item, path, "unsupported_value_type")
        converted = convert(value)
        content = (json.dumps(converted, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(content) > COMPONENT_IO_LIMITS["max_bytes"]:
            converted = summary(value, "$", "artifact_exceeds_byte_limit")
            content = (json.dumps(converted, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        return content, omissions, redacted

    def _component_artifact(self, row, field, value, directory):
        row[field + "_path"], row[field + "_sha256"] = None, None
        row[field + "_not_recorded_reason"] = "observer_io_error"
        content, omissions, redacted = self._component_payload(value)
        target = directory / (field + ".json")
        self._write(target, content, exclusive=True)
        row.update({field + "_path": str(target), field + "_sha256": _hash(content),
                    field + "_recording": "bounded" if omissions else "full",
                    field + "_omissions": omissions, field + "_limits": dict(COMPONENT_IO_LIMITS),
                    field + "_redacted": redacted})
        row.pop(field + "_not_recorded_reason", None)

    def begin_component(self, *, component_name, slot, index, function, inputs):
        call_id = uuid.uuid4().hex
        token = self.component_context.set({"component_call_id": call_id,
                                            "component_name": component_name, "slot": slot, "index": index})
        row = {"schema_version": 1, "timestamp": _utc(), "source_kind": "captured_execution",
               "scope": "unknown", "task_id": None, "attempt_id": None, "question_id": None,
               "status": "running", "seconds": None, "error_type": None,
               "source": None, "skill_hash": None, "output_path": None, "output_sha256": None,
               "output_not_recorded_reason": "component_in_progress", **self._context()}
        directory = self.root / "components" / call_id
        self._observe("component_skill_hash", lambda: row.update(skill_hash=self.skill_hash()))
        self._observe("component_source", lambda: row.update(source=self._callable_source(function, component_name)))
        self._observe("component_inputs", lambda: self._component_artifact(row, "inputs", inputs, directory))
        self._observe("append_component_running", lambda: self._append("component-calls.jsonl", row))
        return row, directory, token, time.monotonic()

    def end_component(self, observation, *, result, error, status):
        if observation is None:
            return
        row, directory, token, started = observation
        try:
            row.update(status=status, seconds=time.monotonic() - started, finished_at=_utc(),
                       error_type=type(error).__name__ if error is not None else None)
            if result is not None:
                self._observe("component_output", lambda: self._component_artifact(row, "output", result, directory))
            else:
                row["output_not_recorded_reason"] = "no_output_available"
            self._observe("append_component_terminal", lambda: self._append("component-calls.jsonl", row))
        finally:
            self.component_context.reset(token)

    def record_compile(self, compiled):
        def record():
            row = {"timestamp": _utc(), "source_kind": "captured_execution", **self._context(),
                   "skill_hash": self.skill_hash()}
            if not hasattr(compiled, "_workflow") or not hasattr(compiled, "_components"):
                row.update(workflow=None, components=[],
                           source_not_recorded_reason="compiled_callable_metadata_unavailable")
                self._append("compile-captures.jsonl", row)
                return
            capture = self._callable_source
            row["workflow"] = capture(compiled._workflow, compiled.workflow_name)
            row["components"] = [{"slot": slot, "index": index,
                                  **capture(function, compiled.binding_names[slot][index])}
                                 for slot, functions in compiled._components.bindings.items()
                                 for index, function in enumerate(functions)]
            self._append("compile-captures.jsonl", row)
        self._observe("compile_sources", record)


class CapturedModel:
    """Delegate once with identical arguments, return object and exception identity."""

    def __init__(self, client, role, recorder):
        self._client, self._role, self._recorder = client, role, recorder

    def __getattr__(self, name):
        return getattr(self._client, name)

    def generate(self, *args, **kwargs):
        observation = self._recorder._observe("start_capture", lambda: self._recorder.start(
            self._client, self._role, args, kwargs))
        started = time.monotonic()
        result, error = None, None
        try:
            result = self._client.generate(*args, **kwargs)
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            if observation is not None:
                seconds = time.monotonic() - started
                self._recorder._observe("finish_capture", lambda: self._recorder.finish(
                    self._client, observation, result, error, seconds))


@contextmanager
def capture_context(model, **fields):
    """Set a worker-local QA/validation scope, otherwise execute unchanged."""
    if isinstance(model, CapturedModel):
        with model._recorder.context(**fields):
            yield
    else:
        yield
