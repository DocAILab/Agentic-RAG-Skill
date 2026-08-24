from pathlib import Path

import pytest

from framework import discover_specs, load_runtime_callable

SKILLS_ROOT = Path(__file__).parents[1] / "framework" / "skills"


class FakeComponents:
    def __init__(self, bindings):
        self.bindings = bindings

    def has(self, slot):
        return bool(self.bindings.get(slot))

    def call(self, slot, inputs, *, index=0):
        return self.bindings[slot][index](inputs)


def _workflow():
    specs = {spec.package_name: spec for spec in discover_specs(SKILLS_ROOT)}
    return load_runtime_callable(specs["agentic-sim-rag"])


def test_rejected_third_round_returns_safe_answer() -> None:
    counter = iter(range(1, 4))
    components = FakeComponents(
        {
            "retriever": [
                lambda inputs: {
                    "documents": [
                        {"id": f"doc-{value}", "text": f"Evidence {value}"}
                        for value in [next(counter)]
                    ]
                }
            ],
            "generator": [lambda inputs: {"answer": "Unsupported candidate"}],
            "critic": [
                lambda inputs: {
                    "approved": False,
                    "score": 0.2,
                    "feedback": "Need more evidence.",
                    "issues": ["Still incomplete."],
                }
            ],
        }
    )

    result = _workflow()({"query": "Question", "top_k": 1}, components)

    assert result["answer"] == "Insufficient evidence to answer reliably."
    assert result["trace"][-1]["reason"] == "max_iterations"
    iterations = [event for event in result["trace"] if event["step"] == "iteration"]
    assert len(iterations) == 3


def test_rejected_round_with_no_new_evidence_returns_safe_answer() -> None:
    document = {"id": "same", "text": "Repeated evidence."}
    critiques = iter(
        [
            {"approved": False, "score": 0.3, "feedback": "More.", "issues": []},
            {"approved": False, "score": 0.3, "feedback": "More.", "issues": []},
        ]
    )
    components = FakeComponents(
        {
            "retriever": [lambda inputs: {"documents": [document]}],
            "generator": [lambda inputs: {"answer": "Candidate"}],
            "critic": [lambda inputs: next(critiques)],
        }
    )

    result = _workflow()({"query": "Question", "max_iterations": 3}, components)

    assert result["answer"] == "Insufficient evidence to answer reliably."
    assert result["trace"][-1] == {
        "step": "stop",
        "iteration": 2,
        "reason": "no_new_evidence",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"query": "   "}, "non-empty query"),
        ({"query": "Question", "top_k": 0}, "top_k must be positive"),
        ({"query": "Question", "max_iterations": 0}, "max_iterations must be positive"),
    ],
)
def test_invalid_iterative_request_is_rejected(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        _workflow()(payload, FakeComponents({}))


@pytest.mark.parametrize("rewriter_result", [{}, {"rewritten_query": "  "}])
def test_empty_rewriter_result_is_rejected(rewriter_result) -> None:
    components = FakeComponents({"rewriter": [lambda inputs: rewriter_result]})

    with pytest.raises(ValueError, match="non-empty rewritten_query"):
        _workflow()({"query": "Question"}, components)


@pytest.mark.parametrize(
    ("critic_result", "message"),
    [
        (
            {"approved": "yes", "score": 1.0, "feedback": "", "issues": []},
            "approved",
        ),
        (
            {"approved": False, "score": 1.5, "feedback": "", "issues": []},
            "score",
        ),
        (
            {"approved": False, "score": 0.5, "feedback": [], "issues": []},
            "feedback",
        ),
        (
            {"approved": False, "score": 0.5, "feedback": "", "issues": [1]},
            "issues",
        ),
    ],
)
def test_invalid_critic_result_is_rejected(critic_result, message) -> None:
    components = FakeComponents(
        {
            "retriever": [lambda inputs: {"documents": []}],
            "generator": [lambda inputs: {"answer": "Candidate"}],
            "critic": [lambda inputs: critic_result],
        }
    )

    with pytest.raises(ValueError, match=message):
        _workflow()({"query": "Question"}, components)
