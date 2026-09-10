"""把 HF 流式数据集确定性切分为多个互斥的小型 RAG task。

与 ``build_task_data``（单文件、按源顺序取前 N 条）不同，本模块先生成
全量 universe manifest（按 ``sha256("{dataset}:{sample_id}")`` 排序，与
``experiments/retrieval`` 的抽样规则一致），再把样本按该顺序切成
互不重叠的 task 文件，并为每个 task 和整档 suite 写入可校验 manifest。

产物布局（默认 triviaqa validation）::

    data/task_flow/triviaqa-validation/
    |-- universe.manifest.json        # 全量有效样本清单（哈希序）
    |-- records.jsonl                 # 规范化全量记录缓存（流式源序）
    |-- k200/
    |   |-- task-0001.jsonl
    |   |-- task-0001.manifest.json
    |   |-- ...
    |   `-- suite.manifest.json
    `-- k20/
        `-- ...

派生文件全部写入本地、由 .gitignore 排除；只有脚本和说明进入仓库。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.retrieval.adapters.common import sample_id
from experiments.retrieval.adapters.triviaqa import adapt_triviaqa
from experiments.retrieval.loading import DatasetItem, iter_huggingface_items
from experiments.retrieval.sampling import _manifest_digest, _sample_digest, write_manifest


class SplitTaskDataError(RuntimeError):
    """表示 task 切分输入、中间状态或产物不符合契约。"""


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "task_flow" / "triviaqa-validation"
DEFAULT_TASK_SIZES = (200,)
DATASET_CHOICES = ("hotpotqa", "2wiki", "2wikimultihopqa", "triviaqa")
TRIVIAQA_RC_REVISION = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
TRIVIAQA_RC_VALIDATION_FILES = tuple(
    f"rc/validation-{index:05d}-of-00004.parquet" for index in range(4)
)


def _disable_hf_ssl_verification() -> None:
    """仅为本进程的数据下载关闭 requests 的 TLS 校验。

    某些网络环境用自签中间证书做 TLS 拦截，Python 默认证书库不信任该
    链；本开关只在用户显式传入 ``--insecure-hf`` 时启用，不修改任何
    全局安装或仓库文件。
    """
    try:
        import requests
        import urllib3
    except ImportError:
        return
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    original = requests.Session.request

    def request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original(self, method, url, **kwargs)

    requests.Session.request = request


def _iter_triviaqa_rc_validation() -> Iterable[DatasetItem]:
    """按固定 revision 直接读取 TriviaQA rc validation 的 4 个 parquet shard。

    Hugging Face 数据集脚本的 builder config 列表近期只暴露 ``default``，
    但仓库中 ``rc/`` 分片仍然存在。这里固定 revision 与文件列表下载到
    本地 HF 缓存后流式读取，保持与仓库现有 adapter/样本格式一致。
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    local_files = [
        hf_hub_download(
            repo_id="mandarjoshi/trivia_qa",
            filename=filename,
            repo_type="dataset",
            revision=TRIVIAQA_RC_REVISION,
        )
        for filename in TRIVIAQA_RC_VALIDATION_FILES
    ]
    rows = load_dataset(
        "parquet",
        data_files={"validation": local_files},
        split="validation",
        streaming=True,
    )
    index = 0
    for row in rows:
        identity = sample_id(row)
        try:
            example = adapt_triviaqa(row)
            yield DatasetItem(index, identity, example=example)
        except (KeyError, TypeError, ValueError) as exc:
            yield DatasetItem(index, identity, error=f"sample {identity}: {exc}")
        index += 1


def _dataset_items(
    dataset: str,
    split: str,
    dataset_config: str | None,
) -> Iterable[DatasetItem]:
    """选择数据源：TriviaQA rc validation 走固定 revision 的 shard 直读。"""
    if dataset == "triviaqa" and split == "validation" and dataset_config in (
        None,
        "rc",
    ):
        yield from _iter_triviaqa_rc_validation()
        return
    yield from iter_huggingface_items(dataset, split, config=dataset_config)


def _record_from_item(
    item: DatasetItem,
    *,
    dataset: str,
) -> dict[str, Any]:
    """把一条有效 DatasetItem 转成与 build_task_data 一致的规范化记录。"""
    example = item.example
    if example is None:
        raise SplitTaskDataError(f"item {item.sample_id!r} has no example")
    return {
        "id": example.id,
        "dataset": example.metadata.get("dataset", dataset),
        "question": example.query,
        "answers": list(example.gold_answers),
        "documents": [document.to_dict() for document in example.documents],
        "relevant_document_ids": list(example.relevant_document_ids),
        "label_type": example.label_type,
    }


def _row_key(
    dataset: str,
    sample_id: str,
    record: Mapping[str, Any],
) -> str:
    """构造唯一且稳定的行键。

    TriviaQA rc 的同一 QuestionId 会以 wiki/web 两份证据变体出现，不能只
    用 sample_id 去重或排序。行键 = question id + 候选文档内容指纹，
    只依赖数据本身，不依赖 shard 顺序。
    """
    encoded = json.dumps(
        record["documents"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{dataset}:{sample_id}:{fingerprint}"


def _is_valid(item: DatasetItem, *, require_labels: bool) -> bool:
    if item.error or item.example is None:
        return False
    if require_labels and not item.example.has_labels:
        return False
    if require_labels and not item.example.gold_answers:
        return False
    return True


def _write_records(
    items: Iterable[DatasetItem],
    records_path: Path,
    *,
    dataset: str,
    require_labels: bool,
) -> tuple[list[tuple[str, str, int]], dict[str, int]]:
    """流式写出规范化记录，返回 (row_key, sample_id, source_index) 列表。"""
    entries: list[tuple[str, str, int]] = []
    stats = {"invalid": 0, "duplicate": 0}
    seen: set[str] = set()
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            if not _is_valid(item, require_labels=require_labels):
                stats["invalid"] += 1
                continue
            record = _record_from_item(item, dataset=dataset)
            key = _row_key(dataset, item.sample_id, record)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            wrapper = {
                "source_index": item.source_index,
                "sample_id": item.sample_id,
                "row_key": key,
                "record": record,
            }
            handle.write(json.dumps(wrapper, ensure_ascii=False) + "\n")
            entries.append((key, item.sample_id, item.source_index))
    return entries, stats


def create_universe(
    items: Iterable[DatasetItem],
    output_root: Path,
    *,
    dataset: str,
    split: str,
    dataset_config: str | None,
    require_labels: bool,
) -> dict[str, Any]:
    """构建全量 universe manifest，并把规范化记录写入本地缓存。"""
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "records.jsonl"
    records_tmp = output_root / "records.jsonl.tmp"
    universe_path = output_root / "universe.manifest.json"
    if records_path.exists() or universe_path.exists() or records_tmp.exists():
        raise SplitTaskDataError(
            f"refusing to overwrite existing universe state: {output_root}"
        )
    try:
        entries, stats = _write_records(
            items,
            records_tmp,
            dataset=dataset,
            require_labels=require_labels,
        )
    except Exception:
        records_tmp.unlink(missing_ok=True)
        raise
    if not entries:
        records_tmp.unlink(missing_ok=True)
        raise SplitTaskDataError("no valid samples found in the dataset split")
    os.replace(records_tmp, records_path)
    ordered = sorted(entries, key=lambda entry: _sample_digest(dataset, entry[0]))
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "order_method": "sha256:{dataset}:{row_key}",
        "require_labels": require_labels,
        "requested_size": len(ordered),
        "source_index_count": len(entries),
        "invalid_count": stats["invalid"],
        "duplicate_rows": stats["duplicate"],
        "selected_keys": [row_key for row_key, _, _ in ordered],
        "sample_ids": [sample_id for _, sample_id, _ in ordered],
        "source_indices": [source_index for _, _, source_index in ordered],
    }
    manifest = {**payload, "digest": _manifest_digest(payload)}
    write_manifest(universe_path, manifest)
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_tier(
    output_root: Path,
    *,
    dataset: str,
    split: str,
    dataset_config: str | None,
    task_size: int,
    force: bool,
) -> dict[str, Any]:
    """把一个 universe 切成 task_size 一档的互斥 task 文件并写 suite。"""
    if task_size < 1:
        raise SplitTaskDataError("task_size must be a positive integer")
    universe_path = output_root / "universe.manifest.json"
    records_path = output_root / "records.jsonl"
    if not universe_path.is_file() or not records_path.is_file():
        raise SplitTaskDataError(
            "universe manifest or records cache is missing; run universe "
            "creation first"
        )
    universe = _read_universe(universe_path, dataset=dataset, split=split)
    selected_keys = list(universe["selected_keys"])
    rank_by_key = {row_key: rank for rank, row_key in enumerate(selected_keys)}
    total = len(selected_keys)
    num_tasks = math.ceil(total / task_size)
    tier_root = output_root / f"k{task_size}"
    suite_path = tier_root / "suite.manifest.json"
    if suite_path.exists() and not force:
        raise SplitTaskDataError(
            f"tier already exists (use --force to rebuild): {suite_path}"
        )
    tier_root.mkdir(parents=True, exist_ok=True)

    task_paths: dict[int, Path] = {}
    handles: dict[int, Any] = {}
    task_sample_ids: list[list[str]] = [[] for _ in range(num_tasks)]
    task_row_keys: list[list[str]] = [[] for _ in range(num_tasks)]
    task_indices: list[list[int]] = [[] for _ in range(num_tasks)]
    try:
        with records_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                wrapper = json.loads(line)
                row_key = wrapper["row_key"]
                if row_key not in rank_by_key:
                    raise SplitTaskDataError(
                        f"records cache contains an id outside universe: "
                        f"{row_key}"
                    )
                task_index = rank_by_key[row_key] // task_size
                if task_index >= num_tasks:
                    raise SplitTaskDataError(
                        f"task index out of range: {row_key} -> {task_index}"
                    )
                handle = handles.get(task_index)
                if handle is None:
                    path = tier_root / f"task-{task_index:04d}.jsonl"
                    task_paths[task_index] = path
                    handle = path.open("w", encoding="utf-8", newline="\n")
                    handles[task_index] = handle
                handle.write(
                    json.dumps(wrapper["record"], ensure_ascii=False) + "\n"
                )
                task_sample_ids[task_index].append(wrapper["sample_id"])
                task_row_keys[task_index].append(row_key)
                task_indices[task_index].append(int(wrapper["source_index"]))
    finally:
        for handle in handles.values():
            handle.close()

    files: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    for task_index in range(num_tasks):
        path = task_paths.get(task_index)
        if path is None:
            raise SplitTaskDataError(f"task file was not created: {task_index}")
        sample_ids = task_sample_ids[task_index]
        row_keys = task_row_keys[task_index]
        all_keys.update(row_keys)
        task_manifest = {
            "schema_version": 1,
            "dataset": dataset,
            "dataset_config": dataset_config,
            "split": split,
            "task_id": f"{task_index:04d}",
            "task_index": task_index,
            "task_size": task_size,
            "examples": len(sample_ids),
            "source_indices": task_indices[task_index],
            "sample_ids": sample_ids,
            "row_keys": row_keys,
            "output": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        manifest_path = path.with_name(f"{path.stem}.manifest.json")
        manifest_path.write_text(
            json.dumps(task_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        files.append(
            {
                "task": path.name,
                "manifest": manifest_path.name,
                "examples": task_manifest["examples"],
                "bytes": task_manifest["bytes"],
                "sha256": task_manifest["sha256"],
            }
        )

    coverage = {
        "universe_samples": total,
        "assigned_rows": len(all_keys),
        "duplicated_rows": 0,
        "missing_rows": total - len(all_keys),
    }
    if len(all_keys) != total or all_keys != set(selected_keys):
        raise SplitTaskDataError(
            f"tier coverage check failed: {coverage}"
        )
    payload = {
        "schema_version": 1,
        "name": f"{dataset}-{split}-k{task_size}-v1",
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "order_method": universe.get("order_method"),
        "universe_manifest": universe_path.name,
        "task_size": task_size,
        "num_tasks": num_tasks,
        "coverage": coverage,
        "files": files,
    }
    suite = {**payload, "digest": _manifest_digest(payload)}
    suite_path.write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return suite


def _read_universe(
    path: Path,
    *,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("digest", None)
    if digest != _manifest_digest(payload):
        raise SplitTaskDataError(f"universe manifest digest mismatch: {path}")
    manifest = {**payload, "digest": digest}
    if manifest.get("dataset") != dataset or manifest.get("split") != split:
        raise SplitTaskDataError(
            "universe manifest dataset/split does not match the request"
        )
    return manifest


def build_task_suite(
    items: Iterable[DatasetItem],
    output_root: Path,
    *,
    dataset: str,
    split: str,
    dataset_config: str | None,
    task_sizes: Sequence[int],
    require_labels: bool,
    force: bool,
) -> list[dict[str, Any]]:
    """执行完整切分流程：universe 构建 + 每个 task_size 档位切分。"""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    universe_path = output_root / "universe.manifest.json"
    records_path = output_root / "records.jsonl"
    if universe_path.is_file() and records_path.is_file():
        universe = _read_universe(universe_path, dataset=dataset, split=split)
        if universe.get("dataset_config") != dataset_config or universe.get(
            "require_labels"
        ) != require_labels:
            raise SplitTaskDataError(
                "existing universe uses different dataset_config/require_labels; "
                "move it away or choose a different output root"
            )
    else:
        if universe_path.exists() or records_path.exists():
            raise SplitTaskDataError(
                f"incomplete universe state in {output_root}; remove it before "
                "rebuilding"
            )
        universe = create_universe(
            items,
            output_root,
            dataset=dataset,
            split=split,
            dataset_config=dataset_config,
            require_labels=require_labels,
        )
    suites = []
    for task_size in sorted(set(int(size) for size in task_sizes)):
        suites.append(
            materialize_tier(
                output_root,
                dataset=dataset,
                split=split,
                dataset_config=dataset_config,
                task_size=task_size,
                force=force,
            )
        )
    return suites


def write_task_flow_yaml(
    suite: Mapping[str, Any],
    output_root: Path,
    yaml_path: Path,
    *,
    config_name: str = "settings.yaml",
) -> None:
    """为一个已生成的 K 档 suite 写出本地 task_flow YAML。"""
    task_size = int(suite["task_size"])
    lines = [
        "# 生成的 TriviaQA 多 Task 自优化配置；按需复制后运行。",
        "# 所有相对路径以当前 self_optimization/ 目录为基准。",
        "schema_version: 1",
        "",
        "self_optimization_config: " + config_name,
        "",
        "workflow:",
        "  # 每个 task 最多向优化模型发送多少条检索过程。",
        "  max_process_examples: 5",
        "  tasks:",
    ]
    for entry in suite["files"]:
        task_file = entry["task"]
        task_index = int(task_file.split("-")[1].split(".")[0])
        lines.append(f'    - name: triviaqa-validation-k{task_size}-{task_index:04d}')
        lines.append("      dataset: triviaqa")
        relative = output_root.joinpath(f"k{task_size}", task_file)
        data_path = os.path.relpath(relative, yaml_path.parent).replace(
            os.sep,
            "/",
        )
        lines.append(f"      data_path: {data_path}")
        lines.append("      request:")
        lines.append("        top_k: 10")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="triviaqa",
        choices=DATASET_CHOICES,
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="universe、records 缓存与各 K 档产物根目录",
    )
    parser.add_argument(
        "--task-size",
        nargs="+",
        type=int,
        default=list(DEFAULT_TASK_SIZES),
        metavar="K",
        help="一档或多档 task 规模，例如 --task-size 200 20",
    )
    parser.add_argument(
        "--no-require-labels",
        action="store_false",
        dest="require_labels",
        help="把无弱标签样本也纳入 universe（默认排除）",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--insecure-hf",
        action="store_true",
        help="网络环境存在 TLS 自签拦截时关闭证书校验（仅限数据下载，不落盘）",
    )
    parser.add_argument(
        "--write-task-flow",
        action="store_true",
        help="为每个生成的 K 档写入 self_optimization/task_flow.triviaqa-kK.yaml",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.insecure_hf:
        _disable_hf_ssl_verification()
    items = _dataset_items(
        args.dataset,
        args.split,
        args.dataset_config,
    )
    suites = build_task_suite(
        items,
        args.output,
        dataset=args.dataset,
        split=args.split,
        dataset_config=args.dataset_config,
        task_sizes=args.task_size,
        require_labels=args.require_labels,
        force=args.force,
    )
    summaries = []
    task_flow_root = REPO_ROOT / "self_optimization"
    for suite in suites:
        summary = {
            "task_size": suite["task_size"],
            "num_tasks": suite["num_tasks"],
            "universe_samples": suite["coverage"]["universe_samples"],
        }
        summaries.append(summary)
        if args.write_task_flow:
            yaml_path = (
                task_flow_root
                / f"task_flow.triviaqa-k{suite['task_size']}.yaml"
            )
            write_task_flow_yaml(
                suite,
                args.output,
                yaml_path,
            )
            summary["task_flow_yaml"] = _repo_relative(yaml_path)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
