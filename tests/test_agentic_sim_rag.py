from pathlib import Path

from framework import SkillKind, discover_specs, load_runtime_callable

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
    return load_runtime_callable(specs["agentic-iterative-rag"])


def test_sim_rag_manifest_declares_iterative_component_slots() -> None:
    specs = {spec.package_name: spec for spec in discover_specs(SKILLS_ROOT)}

    spec = specs["agentic-iterative-rag"]
    slots = {slot.name: slot for slot in spec.slots}

    assert spec.kind is SkillKind.AGENTIC
    assert spec.runtime_id == "agentic.iterative.sim_rag"
    assert set(slots) == {
        "rewriter",
        "retriever",
        "reranker",
        "generator",
        "critic",
    }
    assert (slots["rewriter"].min_count, slots["rewriter"].max_count) == (0, 1)
    assert (slots["retriever"].min_count, slots["retriever"].max_count) == (1, 1)
    assert (slots["reranker"].min_count, slots["reranker"].max_count) == (0, 1)
    assert (slots["generator"].min_count, slots["generator"].max_count) == (1, 1)
    assert (slots["critic"].min_count, slots["critic"].max_count) == (1, 1)


def test_first_round_critic_approval_returns_candidate_answer() -> None:
    document = {"id": "orchard", "text": "Apple trees grow in orchards."}
    components = FakeComponents(
        {
            "retriever": [lambda inputs: {"documents": [document]}],
            "generator": [lambda inputs: {"answer": "In orchards."}],
            "critic": [
                lambda inputs: {
                    "approved": True,
                    "score": 0.95,
                    "feedback": "Supported by the evidence.",
                    "issues": [],
                }
            ],
        }
    )

    result = _workflow()(
        {"query": "Where do apples grow?", "documents": [document], "top_k": 2},
        components,
    )

    assert result["answer"] == "In orchards."
    assert result["documents"] == [document]
    assert result["trace"][0]["iteration"] == 1
    assert result["trace"][0]["critic"]["approved"] is True
    assert result["trace"][-1] == {
        "step": "stop",
        "iteration": 1,
        "reason": "critic_approved",
    }
    assert [call[0] for call in components.calls] == [
        "retriever",
        "generator",
        "critic",
    ]


def test_rejected_round_uses_feedback_query_and_accumulates_unique_evidence() -> None:
    first = {"id": "a", "text": "Initial evidence."}
    second = {"id": "b", "text": "Missing evidence."}
    retrievals = iter(
        [
            {"documents": [first]},
            {"documents": [first, second]},
        ]
    )
    critiques = iter(
        [
            {
                "approved": False,
                "score": 0.4,
                "feedback": "Find the missing location.",
                "issues": ["Location is unsupported."],
            },
            {
                "approved": True,
                "score": 0.9,
                "feedback": "Now supported.",
                "issues": [],
            },
        ]
    )
    components = FakeComponents(
        {
            "retriever": [lambda inputs: next(retrievals)],
            "generator": [
                lambda inputs: {"answer": f"Answer with {len(inputs['documents'])} docs"}
            ],
            "critic": [lambda inputs: next(critiques)],
        }
    )

    result = _workflow()(
        {"query": "Original question", "top_k": 2, "max_iterations": 3},
        components,
    )

    retriever_calls = [inputs for slot, inputs in components.calls if slot == "retriever"]
    generator_calls = [inputs for slot, inputs in components.calls if slot == "generator"]
    assert [call["top_k"] for call in retriever_calls] == [2, 4]
    assert "Location is unsupported." in retriever_calls[1]["query"]
    assert "Find the missing location." not in retriever_calls[1]["query"]
    assert [doc["id"] for doc in generator_calls[1]["documents"]] == ["a", "b"]
    assert all(call["query"] == "Original question" for call in generator_calls)
    assert result["answer"] == "Answer with 2 docs"
    assert [doc["id"] for doc in result["documents"]] == ["a", "b"]
    assert [event["new_document_count"] for event in result["trace"][:-1]] == [1, 1]


def test_hyde_rewrites_only_retrieval_and_reranker_uses_original_query() -> None:
    document = {"id": "evidence", "text": "Grounded source text."}
    components = FakeComponents(
        {
            "rewriter": [
                lambda inputs: {"rewritten_query": f"hypothesis::{inputs['query']}"}
            ],
            "retriever": [lambda inputs: {"documents": [document]}],
            "reranker": [lambda inputs: {"documents": inputs["documents"]}],
            "generator": [lambda inputs: {"answer": "Grounded answer."}],
            "critic": [
                lambda inputs: {
                    "approved": True,
                    "score": 1.0,
                    "feedback": "Grounded.",
                    "issues": [],
                }
            ],
        }
    )

    result = _workflow()({"query": "Original question", "top_k": 3}, components)

    calls = {slot: inputs for slot, inputs in components.calls}
    assert calls["rewriter"]["query"] == "Original question"
    assert calls["retriever"]["query"] == "hypothesis::Original question"
    assert calls["reranker"]["query"] == "Original question"
    assert calls["generator"]["query"] == "Original question"
    assert calls["critic"]["query"] == "Original question"
    assert calls["generator"]["documents"] == [document]
    assert result["documents"] == [document]
    assert result["trace"][0]["retrieval_query"] == "hypothesis::Original question"
