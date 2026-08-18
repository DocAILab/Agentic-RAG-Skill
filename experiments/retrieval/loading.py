"""Hugging Face 流式数据加载与逐样本错误隔离。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .adapters import AdapterError, adapt_hotpotqa, adapt_triviaqa, adapt_two_wiki
from .adapters.common import sample_id
from .schema import RetrievalExample

Adapter = Callable[[Mapping[str, Any]], RetrievalExample]

DATASETS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", adapt_hotpotqa),
    "2wikimultihopqa": ("xanhho/2WikiMultihopQA", None, adapt_two_wiki),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc", adapt_triviaqa),
}
TWO_WIKI_FILES = {
    split: f"https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/{split}.parquet"
    for split in ("train", "dev", "test")
}


@dataclass(frozen=True, slots=True)
class DatasetItem:
    source_index: int
    sample_id: str
    example: RetrievalExample | None = None
    error: str | None = None


def iter_huggingface_items(
    dataset: str,
    split: str,
    *,
    config: str | None = None,
    load_dataset_fn=None,
) -> Iterator[DatasetItem]:
    """流式加载指定 split，并把坏样本转换为可记录的 DatasetItem。"""
    dataset_key = _dataset_key(dataset)
    path, default_config, adapter = DATASETS[dataset_key]
    loader = load_dataset_fn or _load_dataset
    resolved_split = "dev" if dataset_key == "2wikimultihopqa" and split == "validation" else split
    rows = _load_rows(
        loader,
        dataset_key=dataset_key,
        path=path,
        config=config if config is not None else default_config,
        split=resolved_split,
    )
    for index, row in enumerate(rows):
        identity = sample_id(row)
        try:
            yield DatasetItem(index, identity, example=adapter(row))
        except AdapterError as exc:
            yield DatasetItem(index, exc.sample_id, error=str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            yield DatasetItem(index, identity, error=f"sample {identity}: {exc}")


def _dataset_key(value: str) -> str:
    normalized = value.strip().lower().replace("-", "")
    aliases = {"2wiki": "2wikimultihopqa", "2wikimultihopqa": "2wikimultihopqa"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DATASETS:
        raise ValueError(f"Unsupported dataset: {value}")
    return normalized


def _load_rows(loader, *, dataset_key, path, config, split):
    if dataset_key == "2wikimultihopqa":
        if split not in TWO_WIKI_FILES:
            raise ValueError(f"Unsupported 2Wiki split: {split}")
        return loader(
            "parquet",
            data_files={split: TWO_WIKI_FILES[split]},
            split=split,
            streaming=True,
        )
    return loader(path, name=config, split=split, streaming=True)


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Retrieval benchmarks require the 'datasets' package; "
            "install the experiment extra"
        ) from exc
    return load_dataset(*args, **kwargs)
