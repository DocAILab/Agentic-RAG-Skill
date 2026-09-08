from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from datasets import Dataset


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = PROJECT_ROOT / "data" / "HotpotQA" / "prepare_hotpotqa.py"


def _load_module():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    spec = importlib.util.spec_from_file_location("hotpotqa_data_loader", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loader = _load_module()
data_builder = importlib.import_module("hotpotqa_data")


def _hotpot_rows(count: int) -> list[dict]:
    return [
        {
            "id": f"id-{index}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "type": "bridge" if index % 2 else "comparison",
            "level": "hard",
            "supporting_facts": {"title": ["Doc"], "sent_id": [0]},
            "context": {
                "title": ["Doc", "Other"],
                "sentences": [["Evidence."], ["Distractor."]],
            },
        }
        for index in range(count)
    ]


def test_cli_defaults_prepare_both_versions_next_to_script() -> None:
    args = loader.parse_args([])

    assert args.version == "all"
    assert args.output_root == SCRIPT_PATH.parent
    assert args.force is False


def test_small_selection_is_stable_when_source_order_changes() -> None:
    forward = [f"id-{index}" for index in range(12)]
    reverse = list(reversed(forward))

    first = {
        forward[index]
        for index in data_builder.select_small_indices(forward, size=5)
    }
    second = {
        reverse[index]
        for index in data_builder.select_small_indices(reverse, size=5)
    }

    assert len(first) == 5
    assert first == second


def test_prepare_small_materializes_fixed_parquet_and_manifest(tmp_path) -> None:
    source = Dataset.from_list(_hotpot_rows(5_001))

    def fake_load_dataset(path, name, *, split, revision):
        assert (path, name, split, revision) == (
            data_builder.REPOSITORY,
            "distractor",
            "train",
            data_builder.REVISION,
        )
        return source

    manifest = data_builder.prepare_small(tmp_path, load_dataset_fn=fake_load_dataset)
    parquet_path = tmp_path / "small" / "train-5000.parquet"

    assert pq.read_table(parquet_path).num_rows == 5_000
    assert manifest["sampling"]["size"] == 5_000
    assert manifest["files"][0]["sha256"] == data_builder.sha256_file(parquet_path)
    assert json.loads((tmp_path / "small" / "manifest.json").read_text()) == manifest


def test_prepare_full_materializes_every_official_configuration_split(tmp_path) -> None:
    source = Dataset.from_list(_hotpot_rows(2))
    configurations = {
        "distractor": {"train": source, "validation": source},
        "fullwiki": {"train": source, "validation": source, "test": source},
    }
    calls = []

    def fake_load_dataset(path, name, *, revision):
        calls.append((path, name, revision))
        return configurations[name]

    manifest = data_builder.prepare_full(tmp_path, load_dataset_fn=fake_load_dataset)

    assert calls == [
        (data_builder.REPOSITORY, "distractor", data_builder.REVISION),
        (data_builder.REPOSITORY, "fullwiki", data_builder.REVISION),
    ]
    assert len(manifest["files"]) == 5
    for configuration, splits in configurations.items():
        for split in splits:
            path = tmp_path / "full" / configuration / f"{split}.parquet"
            assert pq.read_table(path).num_rows == 2


def test_prepare_small_reuses_a_valid_materialized_version(tmp_path) -> None:
    source = Dataset.from_list(_hotpot_rows(5_000))
    first = data_builder.prepare_small(
        tmp_path,
        load_dataset_fn=lambda *args, **kwargs: source,
    )

    def unexpected_loader(*args, **kwargs):
        raise AssertionError("valid local data should not be downloaded again")

    second = data_builder.prepare_small(tmp_path, load_dataset_fn=unexpected_loader)

    assert second == first


def test_corrupt_local_version_requires_force_before_rebuild(tmp_path) -> None:
    source = Dataset.from_list(_hotpot_rows(5_000))

    def fake_loader(*args, **kwargs):
        return source

    data_builder.prepare_small(tmp_path, load_dataset_fn=fake_loader)
    parquet_path = tmp_path / "small" / "train-5000.parquet"
    with parquet_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(data_builder.DatasetStateError, match="--force"):
        data_builder.prepare_small(tmp_path, load_dataset_fn=fake_loader)

    manifest = data_builder.prepare_small(
        tmp_path,
        force=True,
        load_dataset_fn=fake_loader,
    )

    assert manifest["files"][0]["sha256"] == data_builder.sha256_file(parquet_path)


def test_main_materializes_requested_version_and_prints_summary(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = Dataset.from_list(_hotpot_rows(5_000))
    monkeypatch.setattr(
        data_builder,
        "_load_dataset",
        lambda *args, **kwargs: source,
    )

    result = loader.main(
        ["--version", "small", "--output-root", str(tmp_path)]
    )
    summary = json.loads(capsys.readouterr().out)

    assert result == 0
    assert summary == {"small": {"files": 1, "records": 5_000}}


def test_prepare_small_rejects_missing_hotpotqa_fields_before_writing(tmp_path) -> None:
    incomplete = Dataset.from_dict(
        {"id": [f"id-{index}" for index in range(5_000)]}
    )

    with pytest.raises(ValueError, match="missing required columns"):
        data_builder.prepare_small(
            tmp_path,
            load_dataset_fn=lambda *args, **kwargs: incomplete,
        )

    assert not (tmp_path / "small").exists()
