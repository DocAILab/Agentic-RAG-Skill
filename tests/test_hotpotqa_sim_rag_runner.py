import json
from dataclasses import replace
from pathlib import Path

from experiments.hotpotqa.scripts.run_sim_rag import run_experiment
from framework import ModelAPIError, load_framework_config

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments/hotpotqa/configs/sim_rag_optimized.example.yaml"
)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=None):
        self.calls.append((prompt, system, temperature, max_tokens))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _config(tmp_path, max_examples):
    config = load_framework_config(CONFIG_PATH)
    demo = replace(
        config.demo,
        max_examples=max_examples,
        result_path=tmp_path / "result.json",
        log_path=tmp_path / "run.log.jsonl",
    )
    return replace(config, demo=demo)


def test_fixed_runner_executes_without_model_based_skill_selection(tmp_path) -> None:
    model = ScriptedModel(
        [
            "Washington State",
            json.dumps(
                {
                    "approved": True,
                    "score": 1.0,
                    "feedback": "Supported.",
                    "issues": [],
                }
            ),
        ]
    )

    report = run_experiment(_config(tmp_path, 1), model=model, verbose=False)

    assert report["schema_version"] == 2
    assert report["status"] == "completed"
    assert report["experiment"]["successful_examples"] == 1
    assert report["experiment"]["bindings"] == {
        "rewriter": [],
        "retriever": ["component-bm25-retriever"],
        "reranker": [],
        "generator": ["component-grounded-generator"],
        "critic": ["component-critic"],
    }
    assert report["summary"]["em"] == 1.0
    assert len(model.calls) == 2
    assert all("LOADED MANAGE SKILL" not in call[0] for call in model.calls)
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == report


def test_fixed_runner_records_failure_and_continues_next_example(tmp_path) -> None:
    model = ScriptedModel(
        [
            ModelAPIError("temporary empty response"),
            "John Alan Lasseter",
            json.dumps(
                {
                    "approved": True,
                    "score": 1.0,
                    "feedback": "Supported.",
                    "issues": [],
                }
            ),
        ]
    )

    report = run_experiment(_config(tmp_path, 2), model=model, verbose=False)

    assert report["experiment"]["successful_examples"] == 1
    assert report["experiment"]["failed_examples"] == 1
    assert [item["id"] for item in report["examples"]] == [
        "5a76a0005542993569682c64"
    ]
    assert report["failures"] == [
        {
            "id": "5a73cf5e55429978a71e909f",
            "error_type": "ModelAPIError",
            "error": "temporary empty response",
        }
    ]
    checkpoint = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["experiment"]["failed_examples"] == 1
