from __future__ import annotations

from experiments.retrieval.loading import DatasetItem
from experiments.retrieval.run_bm25f_tuning import run_tuning
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample
from experiments.retrieval.selection import bm25f_grid, select_bm25f_defaults


def _record(parameters, all_support, recall=0.9, mrr=0.8):
    metrics = {
        "all_support@5": all_support,
        "recall@5": recall,
        "recall@10": recall,
        "mrr": mrr,
    }
    return {
        "parameters": parameters,
        "metrics_by_dataset": {
            "hotpotqa": dict(metrics),
            "2wiki": dict(metrics),
        },
    }


def test_bm25f_grid_is_complete_unique_and_contains_current_defaults() -> None:
    grid = bm25f_grid()

    assert len(grid) == 72
    assert len({tuple(sorted(item.items())) for item in grid}) == 72
    assert {
        "k1": 1.5,
        "b": 0.75,
        "title_b": 0.75,
        "title_boost": 1.5,
    } in grid


def test_selection_rejects_recall_regression_and_uses_tie_breakers() -> None:
    defaults = {"k1": 1.5, "b": 0.75, "title_b": 0.75, "title_boost": 1.5}
    rejected = {"k1": 2.0, "b": 0.5, "title_b": 0.0, "title_boost": 3.0}
    selected = {"k1": 1.2, "b": 0.75, "title_b": 0.5, "title_boost": 2.0}
    records = [
        _record(defaults, 0.70, recall=0.90),
        _record(rejected, 0.75, recall=0.89),
        _record(selected, 0.749, recall=0.92),
    ]

    result = select_bm25f_defaults(records)

    assert result["selected_parameters"] == selected
    assert result["rejected_for_recall"] == [rejected]


def test_tuning_runner_resumes_completed_configurations(tmp_path) -> None:
    example = RetrievalExample(
        id="one",
        query="orchid",
        documents=(
            RetrievalDocument("gold", "orchid", "text"),
            RetrievalDocument("noise", "noise", "text"),
        ),
        relevant_document_ids=("gold",),
        label_type="supporting_facts",
    )
    items = [DatasetItem(0, "one", example=example)]
    defaults = {"k1": 1.5, "b": 0.75, "title_b": 0.75, "title_boost": 1.5}
    alternative = {"k1": 1.2, "b": 0.5, "title_b": 0.0, "title_boost": 1.0}

    first = run_tuning(
        {"hotpotqa": items, "2wiki": items},
        output_dir=tmp_path,
        configurations=[defaults, alternative],
    )
    second = run_tuning(
        {"hotpotqa": items, "2wiki": items},
        output_dir=tmp_path,
        configurations=[defaults, alternative],
    )

    assert first == second
    assert first["selected_parameters"] == defaults
    assert len((tmp_path / "bm25f_grid.jsonl").read_text().splitlines()) == 2
