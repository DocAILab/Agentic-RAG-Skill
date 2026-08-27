from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from framework import load_framework_config, run_demo
from framework.demo import _sample_batch_queries

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


class DemoGenerationEvaluator:
    """为 demo 测试返回固定生成指标，避免加载 GPT-2。"""

    def evaluate(self, prediction, references):
        """返回可用于验证报告结构的全一生成指标。"""
        return {
            "chrf": 1.0,
            "chrf++": 1.0,
            "meteor": 1.0,
            "r1": 1.0,
            "r2": 1.0,
            "rl": 1.0,
            "ppl": 1.0,
            "cer": 1.0,
            "wer": 1.0,
        }


def test_run_demo_uses_configured_data_and_writes_report(tmp_path, capsys) -> None:
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
                    "selected_agentic_skill": "agentic-sequential-skill",
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
        max_examples=1,
    )
    config = replace(config, demo=demo)

    report = run_demo(
        config,
        model=model,
        generation_evaluator=DemoGenerationEvaluator(),
        verbose=True,
    )

    assert report["summary"] == {
        "count": 1,
        "retrieval": {
            "F1@1": 2 / 3,
            "F1": 0.5,
            "MRR": 1.0,
            "Hit@1": 1.0,
            "Hit@10": 1.0,
            "MAP": 0.8333333333333333,
            "NDCG": 0.9197207891481876,
            "DCG": 1.5,
            "IDCG": 1.6309297535714575,
        },
        "generation": {
            "ChrF": 1.0,
            "ChrF++": 1.0,
            "METEOR": 1.0,
            "R1": 1.0,
            "R2": 1.0,
            "RL": 1.0,
            "PPL": 1.0,
            "CER": 1.0,
            "WER": 1.0,
        },
    }
    assert report["schema_version"] == 2
    assert report["examples"][0]["id"] == first_example["id"]
    assert report["examples"][0]["selection"]["agentic_skill"] == (
        "agentic-sequential-skill"
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
    terminal_output = capsys.readouterr().out
    assert 'Metrics: {"retrieval": {"F1@1": 0.6666666666666666' in terminal_output
    assert (
        'Running Summary: {"count": 1, "retrieval": '
        '{"F1@1": 0.6666666666666666' in terminal_output
    )


def test_run_demo_can_select_once_and_reuse_pipeline_for_batch(tmp_path) -> None:
    """验证批次模式只选择和编译一次，并为多题复用同一 RAG 命令。"""
    config = load_framework_config(CONFIG_PATH)
    assert config.demo is not None
    examples = [
        json.loads(line)
        for line in config.demo.test_path.read_text(encoding="utf-8").splitlines()[:2]
    ]
    model = DemoScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one reusable route.",
                    "reason": "One workflow must serve the batch.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-sequential-skill",
                    "reason": "Use a reusable sequential route.",
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
                    "reason": "Use one component binding for the batch.",
                }
            ),
            examples[0]["answer"],
            examples[1]["answer"],
        ]
    )
    demo = replace(
        config.demo,
        result_path=tmp_path / "batch_results.json",
        log_path=tmp_path / "batch.log.jsonl",
        max_examples=2,
        select_skills_per_example=False,
    )
    config = replace(config, demo=demo)

    report = run_demo(
        config,
        model=model,
        generation_evaluator=DemoGenerationEvaluator(),
        verbose=False,
    )

    assert report["summary"]["count"] == 2
    assert report["batch_selection"]["agentic_skill"] == "agentic-sequential-skill"
    assert len(model.calls) == 5
    assert '"query_count": 2' in model.calls[0][0]
    assert '"sampled_query_count": 2' in model.calls[0][0]
    assert examples[0]["question"] in model.calls[0][0]
    assert examples[1]["question"] in model.calls[0][0]
    assert report["examples"][0]["selection"] == report["examples"][1]["selection"]
    events = [
        json.loads(line)
        for line in demo.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events].count("manage_completed") == 1
    assert [event["event"] for event in events].count("agentic_selected") == 1
    assert [event["event"] for event in events].count("components_selected") == 1
    assert [event["event"] for event in events].count("command_compiled") == 1
    assert next(
        event for event in events if event["event"] == "manage_completed"
    )["selection_scope"] == "batch"


def test_sample_batch_queries_is_uniform_and_limited() -> None:
    """验证批次决策默认可均匀覆盖问题集，而不会发送全部问题文本。"""
    examples = [
        {"question": f"Question {index}"}
        for index in range(100)
    ]

    sampled = _sample_batch_queries(examples, 20)

    assert len(sampled) == 20
    assert sampled[0] == "Question 0"
    assert sampled[-1] == "Question 99"
    assert "Question 1" not in sampled
    assert sampled == _sample_batch_queries(examples, 20)
