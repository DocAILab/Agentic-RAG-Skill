from pathlib import Path

import pytest

from framework import discover_specs, load_runtime_callable

SKILLS_ROOT = Path(__file__).parents[1] / "framework" / "skills"


class FakeComponents:
    def __init__(self, bindings):
        self.bindings = bindings
        self.calls = []

    def has(self, slot):
        return bool(self.bindings.get(slot))

    def call(self, slot, inputs, *, index=0):
        self.calls.append((slot, dict(inputs)))
        return self.bindings[slot][index](inputs)


def _workflow():
    specs = {spec.package_name: spec for spec in discover_specs(SKILLS_ROOT)}
    return load_runtime_callable(specs["agentic-sim-rag"])


def test_approved_abstention_continues_until_direct_answer() -> None:
    retrievals = iter(
        [
            {"documents": [{"id": "a", "text": "Partial evidence."}]},
            {"documents": [{"id": "b", "text": "Paris is the answer."}]},
        ]
    )
    answers = iter(["Insufficient evidence to answer reliably.", "Paris"])
    components = FakeComponents(
        {
            "retriever": [lambda inputs: next(retrievals)],
            "generator": [lambda inputs: {"answer": next(answers)}],
            "critic": [
                lambda inputs: {
                    "approved": True,
                    "score": 0.9,
                    "feedback": "The response is honest.",
                    "issues": [],
                }
            ],
        }
    )

    result = _workflow()(
        {"query": "What is the answer?", "max_iterations": 2},
        components,
    )

    assert result["answer"] == "Paris"
    iterations = [item for item in result["trace"] if item["step"] == "iteration"]
    assert len(iterations) == 2
    assert iterations[0]["critic"]["approved"] is False
    assert "direct answer" in iterations[0]["critic"]["issues"][0].lower()


@pytest.mark.parametrize("critic_max_tokens", [None, 2048])
def test_critic_uses_independent_token_budget(critic_max_tokens) -> None:
    document = {"id": "support", "text": "Supported."}
    components = FakeComponents(
        {
            "retriever": [lambda inputs: {"documents": [document]}],
            "generator": [lambda inputs: {"answer": "Answer"}],
            "critic": [
                lambda inputs: {
                    "approved": True,
                    "score": 1.0,
                    "feedback": "Supported.",
                    "issues": [],
                }
            ],
        }
    )
    request = {"query": "Question", "max_tokens": 256}
    if critic_max_tokens is not None:
        request["critic_max_tokens"] = critic_max_tokens

    _workflow()(request, components)

    calls = {slot: inputs for slot, inputs in components.calls}
    assert calls["generator"]["max_tokens"] == 256
    assert calls["critic"]["max_tokens"] == (critic_max_tokens or 4096)


def test_invalid_critic_token_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="critic_max_tokens must be positive"):
        _workflow()({"query": "Question", "critic_max_tokens": 0}, FakeComponents({}))


def test_follow_up_query_prioritizes_and_bounds_missing_evidence() -> None:
    retrievals = iter(
        [
            {"documents": [{"id": "a", "text": "Partial."}]},
            {"documents": [{"id": "b", "text": "Complete."}]},
        ]
    )
    critiques = iter(
        [
            {
                "approved": False,
                "score": 0.2,
                "feedback": "verbose feedback " * 100,
                "issues": [
                    "Missing person A's occupation.",
                    "Missing person B's occupation.",
                    "Missing relation proving both occupations match.",
                    "This fourth issue must not be included.",
                ],
            },
            {"approved": True, "score": 1.0, "feedback": "OK", "issues": []},
        ]
    )
    components = FakeComponents(
        {
            "retriever": [lambda inputs: next(retrievals)],
            "generator": [lambda inputs: {"answer": "Poet"}],
            "critic": [lambda inputs: next(critiques)],
        }
    )

    _workflow()({"query": "Shared occupation?"}, components)

    retrieval_calls = [value for slot, value in components.calls if slot == "retriever"]
    follow_up = retrieval_calls[1]["query"]
    assert follow_up.startswith("Shared occupation?")
    assert "person A's occupation" in follow_up
    assert "both occupations match" in follow_up
    assert "fourth issue" not in follow_up
    assert "verbose feedback" not in follow_up
    assert len(follow_up) <= 550


def test_iteration_trace_exposes_document_ids_and_new_ids() -> None:
    document = {"id": "support", "text": "Supported."}
    components = FakeComponents(
        {
            "retriever": [lambda inputs: {"documents": [document]}],
            "generator": [lambda inputs: {"answer": "Answer"}],
            "critic": [
                lambda inputs: {
                    "approved": True,
                    "score": 1.0,
                    "feedback": "Supported.",
                    "issues": [],
                }
            ],
        }
    )

    result = _workflow()({"query": "Question"}, components)

    assert result["trace"][0]["document_ids"] == ["support"]
    assert result["trace"][0]["new_document_ids"] == ["support"]
