"""Bounded, measured QA admission within a fixed Task plan and Skill revision.

Concurrency controls admission only. Running model requests are never killed or
silently retried here; transport retry/deadline policy remains with the client.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time


@dataclass(frozen=True, slots=True)
class QAExecutionConfig:
    mode: str = "serial"
    min_workers: int = 1
    initial_workers: int = 2
    max_workers: int = 10
    adjustment_window: int = 4
    throughput_tolerance: float = 0.05
    reuse_healthy_start: bool = True

    def __post_init__(self):
        if not isinstance(self.mode, str) or self.mode not in {"serial", "fixed", "adaptive"}:
            raise ValueError("qa_execution.mode must be serial, fixed or adaptive")
        for key in ("min_workers", "initial_workers", "max_workers", "adjustment_window"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"qa_execution.{key} must be a positive integer")
        if not self.min_workers <= self.initial_workers <= self.max_workers:
            raise ValueError("qa_execution requires min_workers <= initial_workers <= max_workers")
        for key in ("throughput_tolerance",):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"qa_execution.{key} must be finite")
        if not 0 <= self.throughput_tolerance < 1:
            raise ValueError("qa_execution requires 0 <= throughput_tolerance < 1")
        if type(self.reuse_healthy_start) is not bool:
            raise ValueError("qa_execution.reuse_healthy_start must be boolean")

    def to_dict(self):
        return asdict(self)


def parse_qa_execution(value):
    if not isinstance(value, Mapping):
        raise ValueError("workflow.qa_execution must be a mapping")
    unknown = set(value) - set(QAExecutionConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown qa_execution keys: {sorted(unknown)}")
    return QAExecutionConfig(**value)


_SIGNALS = ContextVar("qa_scheduling_signals", default=None)


@contextmanager
def collect_question_signals():
    signals = {"model_calls": 0, "statistics_covered_calls": 0, "retries": 0,
               "failed_attempts": 0, "pressure": False}
    token = _SIGNALS.set(signals)
    try:
        yield signals
    finally:
        _SIGNALS.reset(token)


class SchedulingModel:
    """Collect call-local pressure, without intercepting or replaying transport."""
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def generate(self, *args, **kwargs):
        signals = _SIGNALS.get()
        error = None
        try:
            return self.client.generate(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            if signals is not None:
                signals["model_calls"] += 1
                if error is not None and getattr(error, "retryable", False):
                    signals["pressure"] = True
                try:
                    statistics = getattr(self.client, "last_request_statistics", None)
                    statistics = statistics() if callable(statistics) else None
                    # Unknown coverage stays unknown, never a synthetic zero.
                    keys = ("requests", "succeeded", "failed", "attempts", "retries")
                    valid = (isinstance(statistics, Mapping)
                             and all(type(statistics.get(k)) is int and statistics[k] >= 0 for k in keys)
                             and statistics["requests"] == 1
                             and statistics["succeeded"] + statistics["failed"] == 1
                             and statistics["attempts"] == statistics["retries"] + 1)
                    if valid:
                        retries = statistics["retries"]
                        signals["statistics_covered_calls"] += 1
                        signals["retries"] += retries
                        signals["failed_attempts"] += statistics["attempts"] - statistics["succeeded"]
                        signals["pressure"] |= retries > 0
                except Exception:
                    pass  # Diagnostic access must not replace a model outcome.


@dataclass(slots=True)
class QuestionOutcome:
    value: object
    signals: dict | None = None


class QuestionExecutionError(RuntimeError):
    def __init__(self, index, cause, completed_results, summary):
        super().__init__(f"QA at index {index} failed ({type(cause).__name__})")
        self.index = index
        self.cause = cause
        self.completed_results = completed_results
        self.summary = summary


class _QuestionFailure(Exception):
    def __init__(self, cause, signals, duration):
        self.cause, self.signals, self.duration = cause, signals, duration


def concurrency_history_key(*, service, workflow, bindings, config, resources):
    """Hash public execution/resource identity, never credentials or QA answers."""
    payload = {"service": service, "workflow": workflow, "bindings": bindings,
               "qa_execution": config.to_dict(), "resources": resources}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


class ConcurrencyWarmStartHistory:
    """Only the immediately preceding durably committed Task may seed admission.

    Throughput measurements are deliberately not carried across Tasks. A tested
    limit is decayed by one; a newly proposed limit and the draining tail cannot
    qualify. Restoring from a checkpoint uses its verified committed report only.
    """
    def __init__(self, committed_tasks=()):
        self._last = None
        if committed_tasks:
            self.commit(committed_tasks[-1], len(committed_tasks))

    def commit(self, task_record, task_index):
        self._last = {"task_id": task_record.get("task_id"), "task_index": task_index,
                      "summary": dict(task_record.get("qa_execution") or {})}

    def request(self, config, *, key_sha256, task_index):
        start = {"enabled": config.reuse_healthy_start, "reused": False,
                 "reason": "no_committed_history", "key_sha256": key_sha256,
                 "source_task_id": None, "source_task_index": None,
                 "task_distance": None, "healthy_limit": None, "decay_steps": 1,
                 "starting_limit": config.initial_workers}
        if not config.reuse_healthy_start:
            start["reason"] = "reuse_disabled"
            return start
        if config.mode != "adaptive":
            start["reason"] = "mode_not_adaptive"
            return start
        if self._last is None:
            return start
        last = self._last
        summary = last["summary"]
        prior_start = summary.get("warm_start") or {}
        evidence = summary.get("warm_start_evidence") or {}
        distance = task_index - last["task_index"]
        start.update(source_task_id=last["task_id"], source_task_index=last["task_index"],
                     task_distance=distance, healthy_limit=evidence.get("healthy_limit"))
        if distance != 1:
            start["reason"] = "history_not_adjacent"
        elif prior_start.get("key_sha256") != key_sha256:
            start["reason"] = ("previous_task_has_no_warm_start_evidence" if not prior_start
                               else "execution_identity_changed")
        elif not evidence.get("eligible"):
            start["reason"] = "previous_task_" + str(evidence.get("reason", "missing_evidence"))
        elif (summary.get("request_statistics_availability") != "available"
              or summary.get("observed_retries") != 0
              or summary.get("observed_failed_attempts") != 0
              or summary.get("failed_examples") != 0
              or summary.get("cancelled_examples") != 0
              or summary.get("completed_examples") != summary.get("submitted_examples")):
            start["reason"] = "previous_task_untrusted_statistics"
        elif (type(evidence.get("healthy_limit")) is not int
              or evidence.get("confirmed_saturated_windows", 0) < 2):
            start["reason"] = "previous_task_missing_evidence"
        else:
            target = min(config.max_workers, 10, max(config.min_workers,
                                                   evidence["healthy_limit"] - 1))
            if target > config.initial_workers:
                start.update(reused=True, reason="committed_healthy_limit_decayed",
                             starting_limit=target)
            else:
                start["reason"] = "decayed_limit_not_above_configured_initial"
        return start


class AdaptiveAdmission:
    """Throughput-first admission with hysteresis and real retry pressure.

    QA latency is recorded for diagnosis only. Two consecutive populated windows
    must lose throughput before a one-step decrease. A plateau holds briefly and
    then probes again. A completed QA reporting actual retries halves admission.
    Draining or above-limit transition windows cannot justify throughput backoff.
    """
    def __init__(self, config, total, now, *, starting_limit=None):
        self.config = config
        self.maximum = min(config.max_workers, max(total, 1))
        self.minimum = min(config.min_workers, self.maximum)
        self.limit = (1 if config.mode == "serial" else self.maximum if config.mode == "fixed"
                      else min(config.initial_workers, self.maximum))
        if config.mode == "adaptive" and starting_limit is not None:
            if type(starting_limit) is not int:
                raise ValueError("Adaptive starting_limit must be an integer")
            self.limit = max(self.minimum, min(starting_limit, self.maximum))
        self.started = now
        self.samples = []
        self.reference = None
        self.probe_from = None
        self.hold_windows = 0
        self.degraded_windows = 0
        self.plateau_windows = 0
        self.healthy_limits = []

    def observe(self, duration, pressure, now, *, saturated=True):
        if self.config.mode != "adaptive":
            return None
        reason, target, measured = None, self.limit, {}
        if pressure:
            target = max(self.minimum, self.limit // 2)
            reason = "model_retry_pressure"
            self.reference, self.probe_from, self.hold_windows = None, None, 2
            self.degraded_windows = self.plateau_windows = 0
            self.healthy_limits.clear()
        elif not saturated:
            # The queue naturally empties at the Task tail. Likewise, requests
            # admitted before a decrease may still exceed the new target. Neither
            # interval measures throughput at the requested steady population.
            self.started, self.samples = now, []
            self.degraded_windows = 0
            return None
        else:
            self.samples.append(duration)
            if len(self.samples) < max(self.config.adjustment_window, self.limit):
                return None
            rate = len(self.samples) / max(now - self.started, 1e-9)
            reference = self.reference
            measured = {"window_completed": len(self.samples), "window_seconds": now - self.started,
                        "throughput_qa_per_second": rate,
                        "reference_throughput_qa_per_second": reference,
                        "mean_qa_seconds": sum(self.samples) / len(self.samples),
                        "window_saturated": True}
            if reference is not None and rate < reference * (1 - self.config.throughput_tolerance):
                self.degraded_windows += 1
                measured["consecutive_degraded_windows"] = self.degraded_windows
                # Keep the healthy reference fixed across both bad windows.
                if self.degraded_windows < 2:
                    reason = "throughput_drop_unconfirmed"
                else:
                    target = max(self.minimum, self.probe_from or self.limit - 1)
                    reason = "sustained_throughput_drop"
                    self.reference, self.probe_from, self.hold_windows = None, None, 2
                    self.degraded_windows = self.plateau_windows = 0
            else:
                self.degraded_windows = 0
                # Smooth healthy measurements instead of anchoring forever to
                # one unusually easy set of questions. Latency never participates.
                self.reference = rate if reference is None else (reference + rate) / 2
                self.probe_from = None
                if self.hold_windows:
                    self.hold_windows -= 1
                    reason = "cooldown"
                elif self.limit >= self.maximum:
                    reason = "at_configured_ceiling"
                elif (reference is not None
                      and rate <= reference * (1 + self.config.throughput_tolerance)
                      and self.plateau_windows == 0):
                    self.plateau_windows = 1
                    reason = "throughput_plateau"
                else:
                    self.plateau_windows = 0
                    self.probe_from = self.limit
                    target, reason = self.limit + 1, "probe_capacity"
        old, self.limit = self.limit, target
        if measured:
            if reason in {"throughput_drop_unconfirmed", "sustained_throughput_drop"}:
                self.healthy_limits.clear()
            else:
                # Evidence is the populated window's old limit, never its probe.
                self.healthy_limits.append(old)
        self.started, self.samples = now, []
        return {"reason": reason, "previous_limit": old, "concurrency_limit": target, **measured}


def execute_questions(items, run_one, config, on_event=None, *, warm_start=None):
    """Return values in input order; drain all running workers before any error.

    No background work survives this function. The caller alone owns Task
    evaluation, checkpointing and Skill mutation after the success barrier.
    """
    if not isinstance(config, QAExecutionConfig):
        raise TypeError("Expected QAExecutionConfig")
    started = time.monotonic()
    items = list(items)
    launch = dict(warm_start or {})
    effective = (launch.get("starting_limit") if config.reuse_healthy_start
                 and config.mode == "adaptive" and launch.get("reused") is True else None)
    controller = AdaptiveAdmission(config, len(items), started, starting_limit=effective)
    effective_initial = controller.limit
    launch.setdefault("enabled", config.reuse_healthy_start)
    launch.setdefault("reused", False)
    launch.setdefault("reason", "no_committed_history")
    for key in ("key_sha256", "source_task_id", "source_task_index", "task_distance", "healthy_limit"):
        launch.setdefault(key, None)
    launch.setdefault("decay_steps", 1)
    launch["starting_limit"] = effective_initial
    if not config.reuse_healthy_start or config.mode != "adaptive":
        launch.update(reused=False, reason="reuse_disabled" if not config.reuse_healthy_start
                      else "mode_not_adaptive")
    completed, pending, failures = {}, {}, []
    next_index = peak = 0
    adjustments, samples = [], []
    totals = {"model_calls": 0, "statistics_covered_calls": 0, "retries": 0, "failed_attempts": 0}
    saw_pressure = False

    def emit(event, **fields):
        if on_event is not None:
            on_event(event, **fields)

    def invoke(index, item):
        before = time.monotonic()
        with collect_question_signals() as signals:
            try:
                result = run_one(index, item)
            except BaseException as exc:
                raise _QuestionFailure(exc, dict(signals), time.monotonic() - before) from exc
            outcome = result if isinstance(result, QuestionOutcome) else QuestionOutcome(result)
            if outcome.signals is None:
                outcome.signals = dict(signals)
            return outcome, time.monotonic() - before

    emit("task_qa_scheduling_started", config=config.to_dict(), concurrency_limit=controller.limit,
         configured_initial_concurrency=config.initial_workers,
         effective_initial_concurrency=effective_initial, warm_start=dict(launch),
         total_examples=len(items))
    # Submit only admitted work rather than queueing all QA in the executor.
    pool = (None if config.mode == "serial" else
            ThreadPoolExecutor(max_workers=controller.maximum, thread_name_prefix="xragos-qa"))
    try:
        while pending or (next_index < len(items) and not failures):
            while not failures and next_index < len(items) and len(pending) < controller.limit:
                index = next_index
                if pool is None:
                    future = Future()
                    try:
                        future.set_result(copy_context().run(invoke, index, items[index]))
                    except BaseException as exc:
                        future.set_exception(exc)
                else:
                    future = pool.submit(copy_context().run, invoke, index, items[index])
                pending[future] = index
                next_index += 1
                peak = max(peak, len(pending))
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Process every finished error before admitting another question.
            for future in sorted(done, key=lambda f: pending[f]):
                index = pending.pop(future)
                if future.cancelled():
                    continue
                try:
                    outcome, duration = future.result()
                    completed[index] = outcome.value
                    signal, status = outcome.signals or {}, "completed"
                except _QuestionFailure as failed:
                    failures.append((index, failed.cause))
                    signal, duration, status = failed.signals, failed.duration, "failed"
                except BaseException as exc:
                    failures.append((index, exc))
                    continue
                for key in totals:
                    totals[key] += int(signal.get(key, 0))
                saw_pressure |= bool(signal.get("pressure"))
                sample = {"example_index": index + 1, "duration_seconds": duration, "status": status,
                          "model_calls": signal.get("model_calls", 0),
                          "statistics_covered_calls": signal.get("statistics_covered_calls", 0),
                          "retries": signal.get("retries") if signal.get("statistics_covered_calls") else None}
                if status == "failed":
                    sample["error_type"] = type(failures[-1][1]).__name__
                samples.append(sample)
                change = (controller.observe(duration, bool(signal.get("pressure")), time.monotonic(),
                                            saturated=next_index < len(items) and len(pending) < controller.limit)
                          if not failures else None)
                if change is not None:
                    change.update(completed_examples=len(completed), in_flight=len(pending))
                    adjustments.append(change)
                    emit("task_qa_concurrency_adjusted", **change)
            if failures:
                for future in pending:
                    future.cancel()
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    coverage = ("available" if totals["model_calls"] and totals["model_calls"] == totals["statistics_covered_calls"]
                else "partial" if totals["statistics_covered_calls"] else "unknown")
    evidence_reason = ("reuse_disabled" if not config.reuse_healthy_start
                       else "mode_not_adaptive" if config.mode != "adaptive"
                       else "qa_failure" if failures
                       else "statistics_not_available" if coverage != "available"
                       else "model_retry_pressure" if saw_pressure or totals["retries"] or totals["failed_attempts"]
                       else "insufficient_healthy_windows" if len(controller.healthy_limits) < 2
                       else "healthy_saturated_windows")
    evidence = {"eligible": evidence_reason == "healthy_saturated_windows", "reason": evidence_reason,
                "healthy_limit": controller.healthy_limits[-1] if controller.healthy_limits else None,
                "confirmed_saturated_windows": len(controller.healthy_limits)}
    summary = {"config": config.to_dict(), "completed_examples": len(completed),
               "configured_initial_concurrency": config.initial_workers,
               "effective_initial_concurrency": effective_initial,
               "warm_start": launch, "warm_start_evidence": evidence,
               "submitted_examples": next_index, "failed_examples": len(failures),
               "failures": [{"example_index": index + 1, "error_type": type(error).__name__}
                            for index, error in failures],
               "cancelled_examples": next_index - len(completed) - len(failures),
               "peak_in_flight": peak, "final_concurrency_limit": controller.limit,
               "wall_seconds": time.monotonic() - started, "adjustments": adjustments,
               "samples": sorted(samples, key=lambda row: row["example_index"]),
               "request_statistics_availability": coverage, "model_calls": totals["model_calls"],
               "statistics_covered_calls": totals["statistics_covered_calls"],
               "observed_retries": totals["retries"] if totals["statistics_covered_calls"] else None,
               "observed_failed_attempts": totals["failed_attempts"] if totals["statistics_covered_calls"] else None,
               "result_order": "input", "adaptation_objective": "measured_qa_throughput",
               "controller_policy": {"name": "throughput_first_v2", "latency_used_for_admission": False,
                                     "decrease_confirmation_windows": 2, "ignore_draining_tail": True}}
    emit("task_qa_scheduling_completed" if not failures else "task_qa_scheduling_failed", **summary)
    if failures:
        index, error = failures[0]
        raise QuestionExecutionError(index, error, completed, summary) from error
    return [completed[index] for index in range(len(items))], summary
