from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from framework import load_framework_config, run_demo

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "framework" / "settings.example.yaml"


class DemoScriptedModel:
    """依次返回三级选择结果和标准答案的 demo 模型替身。"""

    def __init__(self, responses):
        """保存响应队列并初始化调用记录。"""
        self.responses = list(responses)
        self.calls = []

    def generate(
        self,
        prompt,
        *,
        system=None,
        temperature=0.0,
        max_tokens=None,
    ):
        """记录模型调用并返回下一条预设响应。"""
        self.calls.append((prompt, system, temperature, max_tokens))
        return self.responses.pop(0)


def test_run_demo_uses_configured_data_and_writes_report(tmp_path) -> None:
    """验证统一入口从配置加载数据、执行一题并保存测评报告。"""
    config = load_framework_config(CONFIG_PATH)
    assert config.demo is not None
    first_example = json.loads(
        config.demo.test_path.read_text(encoding="utf-8").splitlines()[0]
    )
    model = DemoScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one lexical route.",
                    "reason": "The demo requests BM25.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-vanilla-rag",
                    "reason": "A sequential route is sufficient.",
                }
            ),
            json.dumps(
                {
                    "component_bindings": {
                        "rewriter": [],
                        "retriever": ["component-bm25-retriever"],
                        "reranker": [],
                        "generator": ["component-grounded-generator"],
                    },
                    "reason": "Use the configured lexical retriever.",
                }
            ),
            first_example["answer"],
        ]
    )
    demo = replace(
        config.demo,
        result_path=tmp_path / "demo_results.json",
        log_path=tmp_path / "demo.log.jsonl",
    )
    config = replace(config, demo=demo)

    report = run_demo(config, model=model, verbose=False)

    assert report["summary"] == {
        "count": 1,
        "hit@1": 1.0,
        "hit@10": 1.0,
        "em": 1.0,
        "f1": 1.0,
    }
    assert report["examples"][0]["id"] == first_example["id"]
    assert report["examples"][0]["selection"]["agentic_skill"] == (
        "agentic-vanilla-rag"
    )
    assert json.loads(demo.result_path.read_text(encoding="utf-8")) == report
    events = [
        json.loads(line)
        for line in demo.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "dataset_loaded",
        "example_started",
        "manage_completed",
        "agentic_selected",
        "components_selected",
        "command_compiled",
        "execution_completed",
        "evaluation_completed",
        "run_completed",
    ]
    assert len({event["run_id"] for event in events}) == 1
    assert events[-1]["summary"] == report["summary"]
    assert len(model.calls) == 4
