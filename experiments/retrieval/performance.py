"""Efficiency measurements kept separate from retrieval-quality selection."""

from __future__ import annotations

import statistics
import time


def benchmark_batch_sizes(
    client_factory,
    batch_sizes,
    texts,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> list[dict]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    payload = list(texts)
    results = []
    for batch_size in batch_sizes:
        client = client_factory(batch_size)
        try:
            for _ in range(warmup):
                client.embed(payload)
            durations = []
            for _ in range(repeats):
                started = time.perf_counter()
                client.embed(payload)
                durations.append(time.perf_counter() - started)
        except (MemoryError, RuntimeError) as exc:
            if not _is_out_of_memory(exc):
                raise
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "oom",
                    "error": str(exc),
                }
            )
            continue
        median_seconds = statistics.median(durations)
        results.append(
            {
                "batch_size": batch_size,
                "status": "ok",
                "median_seconds": median_seconds,
                "texts_per_second": (
                    len(payload) / median_seconds if median_seconds > 0 else None
                ),
            }
        )
    return results


def _is_out_of_memory(exc):
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
