"""Bounded, measured QA admission within a fixed Task plan and Skill revision.

Concurrency controls admission only. Running model requests are never killed or
silently retried here; transport retry/deadline policy remains with the client.
"""
from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass
import math
import time


@dataclass(frozen=True, slots=True)
class QAExecutionConfig:
    mode: str = "serial"
    min_workers: int = 1
    initial_workers: int = 2
    max_workers: int = 8
    adjustment_window: int = 4
    throughput_tolerance: float = 0.05
    latency_factor: float = 2.0

    def __post_init__(self):
        if self.mode not in {"serial", "fixed", "adaptive"}:
            raise ValueError("qa_execution.mode must be serial, fixed or adaptive")
        for key in ("min_workers", "initial_workers", "max_workers", "adjustment_window"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"qa_execution.{key} must be a positive integer")
        if not self.min_workers <= self.initial_workers <= self.max_workers:
            raise ValueError("qa_execution requires min_workers <= initial_workers <= max_workers")
        for key in ("throughput_tolerance", "latency_factor"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"qa_execution.{key} must be finite")
        if not 0 <= self.throughput_tolerance < 1 or self.latency_factor <= 1:
            raise ValueError("qa_execution requires 0 <= throughput_tolerance < 1 and latency_factor > 1")

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
                    if isinstance(statistics, Mapping) and type(statistics.get("retries")) is int:
                        retries = max(0, statistics["retries"])
                        attempts = statistics.get("attempts")
                        succeeded = statistics.get("succeeded")
                        signals["statistics_covered_calls"] += 1
                        signals["retries"] += retries
                        if type(attempts) is int and type(succeeded) is int:
                            signals["failed_attempts"] += max(0, attempts - succeeded)
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


class AdaptiveAdmission:
    """Probe one extra worker; retain increases only while throughput improves.

    A retry halves admission immediately. Unproductive probes return to the last
    level, then hold for two windows before probing again. Latency inflation also
    backs off. These are workload heuristics, not a GPU utilization measurement.
    """
    def __init__(self, config, total, now):
        self.config = config
        self.maximum = min(config.max_workers, max(total, 1))
        self.minimum = min(config.min_workers, self.maximum)
        self.limit = (1 if config.mode == "serial" else self.maximum if config.mode == "fixed"
                      else min(config.initial_workers, self.maximum))
        self.started = now
        self.samples = []
        self.reference = None
        self.probe_from = None
        self.hold_windows = 0

    def observe(self, duration, pressure, now):
        if self.config.mode != "adaptive":
            return None
        self.samples.append(duration)
        reason, target, measured = None, self.limit, {}
        if pressure:
            target = max(self.minimum, self.limit // 2)
            reason = "model_retry_pressure"
            self.reference, self.probe_from, self.hold_windows = None, None, 2
        elif len(self.samples) >= max(self.config.adjustment_window, self.limit):
            rate = len(self.samples) / max(now - self.started, 1e-9)
            latency = sum(self.samples) / len(self.samples)
            measured = {"window_completed": len(self.samples), "window_seconds": now - self.started,
                        "throughput_qa_per_second": rate, "mean_qa_seconds": latency}
            if self.reference is not None and self.probe_from is not None:
                previous_rate, previous_latency = self.reference
                if latency > previous_latency * self.config.latency_factor:
                    target, reason = max(self.minimum, self.probe_from), "latency_inflation"
                    self.hold_windows = 2
                elif rate <= previous_rate * (1 + self.config.throughput_tolerance):
                    target, reason = max(self.minimum, self.probe_from), "probe_no_throughput_gain"
                    self.hold_windows = 2
                else:
                    reason = "throughput_improved"
                self.probe_from = None
            elif self.reference is not None and latency > self.reference[1] * self.config.latency_factor:
                target, reason = max(self.minimum, self.limit // 2), "latency_inflation"
                self.hold_windows = 2
            if target == self.limit:
                self.reference = (rate, latency)
                if self.hold_windows:
                    self.hold_windows -= 1
                    reason = reason or "cooldown"
                elif self.limit < self.maximum:
                    self.probe_from = self.limit
                    target, reason = self.limit + 1, "probe_capacity"
                else:
                    reason = reason or "at_configured_ceiling"
            else:
                self.reference = None
            reason = reason or "hold"
        if reason is None:
            return None
        old, self.limit = self.limit, target
        self.started, self.samples = now, []
        return {"reason": reason, "previous_limit": old, "concurrency_limit": target, **measured}


def execute_questions(items, run_one, config, on_event=None):
    """Return values in input order; drain all running workers before any error.

    No background work survives this function. The caller alone owns Task
    evaluation, checkpointing and Skill mutation after the success barrier.
    """
    if not isinstance(config, QAExecutionConfig):
        raise TypeError("Expected QAExecutionConfig")
    started = time.monotonic()
    items = list(items)
    controller = AdaptiveAdmission(config, len(items), started)
    completed, pending, failures = {}, {}, []
    next_index = peak = 0
    adjustments, samples = [], []
    totals = {"model_calls": 0, "statistics_covered_calls": 0, "retries": 0, "failed_attempts": 0}

    def emit(event, **fields):
        if on_event is not None:
            on_event(event, **fields)

    def invoke(index, item):
        before = time.monotonic()
        result = run_one(index, item)
        outcome = result if isinstance(result, QuestionOutcome) else QuestionOutcome(result)
        return outcome, time.monotonic() - before

    emit("task_qa_scheduling_started", config=config.to_dict(), concurrency_limit=controller.limit,
         total_examples=len(items))
    # Submit only admitted work rather than queueing all QA in the executor.
    pool = ThreadPoolExecutor(max_workers=controller.maximum, thread_name_prefix="xragos-qa")
    try:
        while pending or (next_index < len(items) and not failures):
            while not failures and next_index < len(items) and len(pending) < controller.limit:
                index = next_index
                pending[pool.submit(copy_context().run, invoke, index, items[index])] = index
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
                except BaseException as exc:
                    failures.append((index, exc))
                    continue
                completed[index] = outcome.value
                signal = outcome.signals or {}
                for key in totals:
                    totals[key] += int(signal.get(key, 0))
                sample = {"example_index": index + 1, "duration_seconds": duration,
                          "model_calls": signal.get("model_calls", 0),
                          "statistics_covered_calls": signal.get("statistics_covered_calls", 0),
                          "retries": signal.get("retries") if signal.get("statistics_covered_calls") else None}
                samples.append(sample)
                change = controller.observe(duration, bool(signal.get("pressure")), time.monotonic())
                if change is not None:
                    change.update(completed_examples=len(completed), in_flight=len(pending))
                    adjustments.append(change)
                    emit("task_qa_concurrency_adjusted", **change)
            if failures:
                for future in pending:
                    future.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    coverage = ("available" if totals["model_calls"] and totals["model_calls"] == totals["statistics_covered_calls"]
                else "partial" if totals["statistics_covered_calls"] else "unknown")
    summary = {"config": config.to_dict(), "completed_examples": len(completed),
               "submitted_examples": next_index, "failed_examples": len(failures),
               "cancelled_examples": next_index - len(completed) - len(failures),
               "peak_in_flight": peak, "final_concurrency_limit": controller.limit,
               "wall_seconds": time.monotonic() - started, "adjustments": adjustments,
               "samples": sorted(samples, key=lambda row: row["example_index"]),
               "request_statistics_availability": coverage, "model_calls": totals["model_calls"],
               "statistics_covered_calls": totals["statistics_covered_calls"],
               "observed_retries": totals["retries"] if totals["statistics_covered_calls"] else None,
               "observed_failed_attempts": totals["failed_attempts"] if totals["statistics_covered_calls"] else None,
               "result_order": "input", "adaptation_objective": "measured_qa_throughput"}
    emit("task_qa_scheduling_completed" if not failures else "task_qa_scheduling_failed", **summary)
    if failures:
        index, error = failures[0]
        raise QuestionExecutionError(index, error, completed, summary) from error
    return [completed[index] for index in range(len(items))], summary
