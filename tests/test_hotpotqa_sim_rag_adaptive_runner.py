import json
from dataclasses import replace
from pathlib import Path

from experiments.hotpotqa.scripts.run_sim_rag_adaptive import (
    run_adaptive_experiment,
)
from framework import ModelAPIError, load_framework_config

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments/hotpotqa/configs/sim_rag_optimized.example.yaml"
)
ADAPTIVE_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments/hotpotqa/configs/sim_rag_adaptive.example.yaml"
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


def _config(tmp_path, max_examples=1, example_max_attempts=1):
    config = load_framework_config(CONFIG_PATH)
    demo = replace(
        config.demo,
        max_examples=max_examples,
        result_path=tmp_path / "adaptive-result.json",
        log_path=tmp_path / "adaptive.log.jsonl",
        request={
            "top_k": 3,
            "max_tokens": 256,
            "selection_max_tokens": 4096,
            "critic_max_tokens": 4096,
            "max_iterations": 3,
            "example_max_attempts": example_max_attempts,
        },
    )
    return replace(config, demo=demo)


class ConstantEmbeddingModel:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _selection(retriever):
    return json.dumps(
        {
            "component_bindings": {
                "rewriter": [],
                "retriever": [retriever],
                "reranker": [],
                "generator": ["component-grounded-generator"],
                "critic": ["component-critic"],
            },
            "reason": f"Choose {retriever} for this question.",
        }
    )


def _approval():
    return json.dumps(
        {
            "approved": True,
            "score": 1.0,
            "feedback": "Supported.",
            "issues": [],
        }
    )


def test_adaptive_runner_selects_components_for_fixed_sim_rag(tmp_path) -> None:
    bindings = {
        "rewriter": [],
        "retriever": ["component-bm25-retriever"],
        "reranker": [],
        "generator": ["component-grounded-generator"],
        "critic": ["component-critic"],
    }
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "component_bindings": bindings,
                    "reason": "Exact entities favor lexical retrieval.",
                }
            ),
            "Washington State",
            _approval(),
        ]
    )

    report = run_adaptive_experiment(
        _config(tmp_path),
        model=model,
        verbose=False,
    )

    assert report["status"] == "completed"
    assert report["experiment"]["selection_mode"] == "adaptive-components"
    assert report["experiment"]["agentic_skill"] == "agentic-sim-rag"
    assert report["examples"][0]["selection"] == {
        "component_bindings": bindings,
        "reason": "Exact entities favor lexical retrieval.",
    }
    assert report["summary"]["em"] == 1.0
    selection_prompt = model.calls[0][0]
    assert "# SIM-RAG-Inspired Iterative RAG" in selection_prompt
    assert "component-bm25-retriever" in selection_prompt
    assert len(model.calls) == 3
    assert model.calls[0][3] == 4096
    assert json.loads(
        (tmp_path / "adaptive-result.json").read_text(encoding="utf-8")
    ) == report


def test_adaptive_runner_can_choose_different_retrievers_per_example(tmp_path) -> None:
    model = ScriptedModel(
        [
            _selection("component-bm25-retriever"),
            "Washington State",
            _approval(),
            _selection("component-vector-retriever"),
            "John Alan Lasseter",
            _approval(),
        ]
    )

    report = run_adaptive_experiment(
        _config(tmp_path, max_examples=2),
        model=model,
        embedding_model=ConstantEmbeddingModel(),
        verbose=False,
    )

    retrievers = [
        item["selection"]["component_bindings"]["retriever"][0]
        for item in report["examples"]
    ]
    assert retrievers == [
        "component-bm25-retriever",
        "component-vector-retriever",
    ]
    assert report["experiment"]["component_selection_counts"]["retriever"] == {
        "component-bm25-retriever": 1,
        "component-vector-retriever": 1,
    }


def test_adaptive_runner_records_component_selection_failures(tmp_path) -> None:
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": ["component-hyde-rewriter"],
                        "retriever": ["component-bm25-retriever"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                        "critic": ["component-critic"],
                    },
                    "reason": "Invalid test choice.",
                }
            )
        ]
    )

    report = run_adaptive_experiment(
        _config(tmp_path),
        model=model,
        verbose=False,
    )

    assert report["experiment"]["successful_examples"] == 0
    assert report["experiment"]["failed_examples"] == 1
    assert report["failures"][0]["stage"] == "selection"
    assert report["failures"][0]["error_type"] == "SelectionError"
    assert "component-vector-retriever" in report["failures"][0]["error"]


def test_adaptive_runner_retries_transient_selection_failure(tmp_path) -> None:
    model = ScriptedModel(
        [
            "not json",
            _selection("component-bm25-retriever"),
            "Washington State",
            _approval(),
        ]
    )

    report = run_adaptive_experiment(
        _config(tmp_path, example_max_attempts=2),
        model=model,
        verbose=False,
    )

    assert report["experiment"]["successful_examples"] == 1
    assert report["experiment"]["failed_examples"] == 0
    assert report["examples"][0]["attempts"] == 2


def test_adaptive_runner_retries_transient_execution_failure(tmp_path) -> None:
    model = ScriptedModel(
        [
            _selection("component-bm25-retriever"),
            ModelAPIError("OpenAI-compatible response contains no text"),
            _selection("component-bm25-retriever"),
            "Washington State",
            _approval(),
        ]
    )

    report = run_adaptive_experiment(
        _config(tmp_path, example_max_attempts=2),
        model=model,
        verbose=False,
    )

    assert report["experiment"]["successful_examples"] == 1
    assert report["experiment"]["failed_examples"] == 0
    assert report["examples"][0]["attempts"] == 2


def test_adaptive_config_enables_embeddings_without_component_constraints() -> None:
    config = load_framework_config(ADAPTIVE_CONFIG_PATH)

    assert config.embedding is not None
    assert config.embedding.model == "BAAI/bge-large-en-v1.5"
    constraints = config.demo.request["constraints"]
    assert config.demo.request["selection_max_tokens"] == 4096
    assert config.demo.request["example_max_attempts"] == 3
    assert constraints["agentic_skill"] == "agentic-sim-rag"
    assert "retriever" not in constraints
    assert "rewriter" not in constraints
    assert "reranker" not in constraints
