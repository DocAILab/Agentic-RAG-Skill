"""Process-wide ownership of the local cross-encoder used by reloadable Skills."""

from __future__ import annotations

from collections.abc import Sequence
import gc
from threading import RLock
from typing import Any


_LOCAL_MODEL_LOCK = RLock()


def local_model_inference_lock():
    """Return the process-wide reentrant lock for local model load/prediction.

    Batch runners may hold this lock around embedding or PPL model operations to
    avoid overlapping GPU activation peaks with reranking. Reranking holds the
    same lock internally, so an outer caller can safely acquire it as well.
    """
    return _LOCAL_MODEL_LOCK


class _CrossEncoderResource:
    """Keep at most one model resident and serialize its entire inference call.

    Group contexts share this owner. A configuration change releases the previous
    model before loading another, and cannot evict a model during prediction.
    Only the model is retained: Skill scripts are still reloaded independently.
    """

    def __init__(self) -> None:
        self._lock = local_model_inference_lock()
        self._key: tuple[str, int | None, str | None] | None = None
        self._encoder: Any = None

    def score(
        self,
        model: str,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        max_length: int | None,
        device: str | None,
    ) -> list[float]:
        key = (model, max_length, device)
        with self._lock:
            if self._key != key:
                if self._encoder is not None:
                    self._encoder = None
                    self._key = None
                    gc.collect()
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "BGE reranker requires the 'sentence-transformers' package. "
                        'Install it with: pip install -e ".[rerank]"'
                    ) from exc
                self._encoder = CrossEncoder(model, max_length=max_length, device=device)
                self._key = key
            # Do not let model-backed tensors or lazy results outlive the lock.
            return [
                float(score)
                for score in self._encoder.predict(
                    pairs, batch_size=batch_size, show_progress_bar=False,
                )
            ]


_CROSS_ENCODER = _CrossEncoderResource()


def score_cross_encoder_pairs(
    model: str,
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = 32,
    max_length: int | None = None,
    device: str | None = None,
) -> list[float]:
    """Score with the shared, bounded model resource across Skill recompilation."""
    batch_size = int(batch_size)
    max_length = None if max_length is None else int(max_length)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive")
    if not pairs:
        return []
    return _CROSS_ENCODER.score(
        model, pairs, batch_size=batch_size, max_length=max_length, device=device,
    )
