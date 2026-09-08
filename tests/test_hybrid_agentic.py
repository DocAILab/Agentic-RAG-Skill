from pathlib import Path

from framework.spec import load_runtime_callable, load_spec

SKILL = Path(__file__).parents[1] / "framework" / "skills" / "agentic" / "agentic-hybrid-rag"


class FakeComponents:
    def __init__(self, bindings):
        self.bindings = bindings
        self.calls = []

    def has(self, slot):
        return bool(self.bindings.get(slot))

    def call(self, slot, inputs, *, index=0):
        self.calls.append((slot, dict(inputs)))
        return self.bindings[slot][index](inputs)


def workflow():
    return load_runtime_callable(load_spec(SKILL))


def test_classifier_complexity_mode_uses_complexity_labels():
    classifier_path = Path(__file__).parents[1] / "framework" / "skills" / "components" / "component-classifier"
    run = load_runtime_callable(load_spec(classifier_path))

    class Context:
        def call_model(self, prompt, **kwargs):
            assert "non-retrieval" in prompt
            assert "multi-step" in prompt
            return '{"route":"multi-step","reason":"needs hops","confidence":0.9}'

    result = run({"query": "Connect these facts", "classification_mode": "complexity"}, Context())

    assert result == {"route": "multi-step", "reason": "needs hops", "confidence": 0.9}


def test_non_retrieval_route_skips_evidence_components():
    components = FakeComponents({
        "classifier": [lambda inputs: {"route": "non-retrieval", "reason": "common knowledge", "confidence": 0.9}],
        "generator": [lambda inputs: {"answer": "Hello."}],
    })

    result = workflow()({"query": "Say hello"}, components)

    assert result["route"] == "non-retrieval"
    assert result["documents"] == []
    assert [slot for slot, _ in components.calls] == ["classifier", "generator"]
    assert components.calls[0][1]["classification_mode"] == "complexity"
    assert components.calls[1][1]["documents"] == []


def test_single_step_route_retrieves_then_generates():
    components = FakeComponents({
        "classifier": [lambda inputs: {"route": "single-step", "reason": "one lookup", "confidence": 0.8}],
        "retriever": [lambda inputs: {"documents": [{"id": "a", "text": "Apple."}]}],
        "generator": [lambda inputs: {"answer": "Apple."}],
    })

    result = workflow()({"query": "What is this?", "documents": [{"id": "a"}]}, components)

    assert result["answer"] == "Apple."
    assert [slot for slot, _ in components.calls] == ["classifier", "retriever", "generator"]
    assert components.calls[-1][1]["query"] == "What is this?"


def test_multi_step_route_stops_after_critic_approval():
    components = FakeComponents({
        "classifier": [lambda inputs: {"route": "multi-step", "reason": "needs hops", "confidence": 0.95}],
        "retriever": [lambda inputs: {"documents": [{"id": "a", "text": "A."}]}],
        "generator": [lambda inputs: {"answer": "A."}],
        "critic": [lambda inputs: {"approved": True, "score": 1.0, "feedback": "", "issues": []}],
    })

    result = workflow()({"query": "Connect the facts", "max_iterations": 2}, components)

    assert result["route"] == "multi-step"
    assert result["answer"] == "A."
    assert [slot for slot, _ in components.calls] == ["classifier", "retriever", "generator", "critic"]
    assert result["trace"][-1]["reason"] == "critic_approved"


def test_multi_step_retrieves_again_after_critic_rejection():
    retrievals = iter([
        {"documents": [{"id": "a", "text": "First fact."}]},
        {"documents": [{"id": "b", "text": "Second fact."}]},
    ])
    critiques = iter([
        {"approved": False, "score": 0.4, "feedback": "Missing the second fact.", "issues": ["second fact is missing"]},
        {"approved": True, "score": 1.0, "feedback": "", "issues": []},
    ])
    components = FakeComponents({
        "classifier": [lambda inputs: {"route": "multi-step", "reason": "needs hops", "confidence": 0.95}],
        "retriever": [lambda inputs: next(retrievals)],
        "generator": [lambda inputs: {"answer": "Both facts."}],
        "critic": [lambda inputs: next(critiques)],
    })

    result = workflow()({"query": "Connect the facts", "max_iterations": 2}, components)

    retrieval_calls = [inputs for slot, inputs in components.calls if slot == "retriever"]
    assert len(retrieval_calls) == 2
    assert "second fact is missing" in retrieval_calls[1]["query"]
    assert result["answer"] == "Both facts."
    assert [event["reason"] for event in result["trace"] if event["step"] == "stop"] == ["critic_approved"]
