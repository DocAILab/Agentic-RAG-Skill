import json
from pathlib import Path

from framework import discover_specs, load_runtime_callable

SKILLS_ROOT = Path(__file__).parents[1] / "framework" / "skills"


class FakeContext:
    def __init__(self):
        self.calls = []

    def call_model(self, prompt, *, temperature=0.0, max_tokens=None):
        self.calls.append((prompt, temperature, max_tokens))
        return json.dumps(
            {
                "approved": False,
                "score": 0.4,
                "feedback": "A required hop is missing.",
                "issues": ["Find the second supporting fact."],
            }
        )


def _critic():
    specs = {spec.package_name: spec for spec in discover_specs(SKILLS_ROOT)}
    return load_runtime_callable(specs["component-critic"])


def test_critic_prompt_requires_direct_complete_multihop_answer() -> None:
    context = FakeContext()

    result = _critic()(
        {
            "query": "What occupation is shared by both people?",
            "answer": "Insufficient evidence.",
            "documents": [{"id": "a", "text": "The first person was a poet."}],
            "max_tokens": 4096,
        },
        context,
    )

    prompt, temperature, max_tokens = context.calls[0]
    assert result["approved"] is False
    assert "direct answer" in prompt
    assert "every required fact or reasoning hop" in prompt
    assert "Abstentions" in prompt
    assert "approved=false" in prompt
    assert "missing evidence targets" in prompt
    assert temperature == 0.0
    assert max_tokens == 4096
