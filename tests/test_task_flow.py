from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from framework import load_framework_config
from self_optimization import (
    OptimizationSettings,
    RAGTaskConfig,
    SelfOptimizationConfig,
    TaskWorkflowConfig,
    load_task_workflow_config,
    run_task_workflow,
)
from self_optimization.task_flow import _compare_revision_metrics

PROJECT_ROOT = Path(__file__).parents[1]
SKILL_ROOT = PROJECT_ROOT / "framework" / "skills"
FRAMEWORK_CONFIG_PATH = PROJECT_ROOT / "framework" / "settings.example.yaml"


class ScriptedModel:
    """按顺序返回预设文本并保存任务流中的全部模型调用。"""

    def __init__(self, responses):
        """保存响应队列和空调用记录。"""
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
        """记录提示参数并返回下一条预设响应。"""
        self.calls.append((prompt, system, temperature, max_tokens))
        return self.responses.pop(0)


class FixedGenerationEvaluator:
    """返回固定生成指标，避免单元测试下载 PPL 模型。"""

    def evaluate(self, prediction, references):
        """为任意任务流预测返回完整的 XRAG 生成指标。"""
        return {
            "chrf": 1.0,
            "chrf++": 1.0,
            "meteor": 1.0,
            "r1": 1.0,
            "r2": 1.0,
            "rl": 1.0,
            "ppl": 1.0,
            "cer": 0.0,
            "wer": 0.0,
        }


def test_revision_metric_gate_requires_generation_improvement_and_gold_answer() -> None:
    """验证生成器修订既要提升指标，也必须包含受监督的正确短答案。"""
    baseline = {
        "retrieval": {"F1": 1.0, "MRR": 1.0, "Hit@10": 1.0, "MAP": 1.0, "NDCG": 1.0},
        "generation": {"ChrF++": 0.0, "METEOR": 0.0, "R1": 0.0, "RL": 0.0},
    }
    revised = {
        "retrieval": dict(baseline["retrieval"]),
        "generation": {"ChrF++": 0.4, "METEOR": 0.3, "R1": 0.5, "RL": 0.5},
    }

    wrong = _compare_revision_metrics(
        selected_skill="component-grounded-generator",
        baseline=baseline,
        revised=revised,
        prediction="The answer is Alasdair Mor.",
        gold_answers=("Domhnall mac Raghnaill",),
    )
    correct = _compare_revision_metrics(
        selected_skill="component-grounded-generator",
        baseline=baseline,
        revised=revised,
        prediction="Domhnall mac Raghnaill",
        gold_answers=("Domhnall mac Raghnaill",),
    )

    assert wrong["improved"] is False
    assert correct["improved"] is True


def test_revision_metric_gate_rejects_generic_manage_change_without_gain() -> None:
    """验证不改变真实输出指标的 Manage 文档修改会被拒绝。"""
    metrics = {
        "retrieval": {"F1": 1.0, "MRR": 1.0, "Hit@10": 1.0, "MAP": 1.0, "NDCG": 1.0},
        "generation": {"ChrF++": 0.4, "METEOR": 0.3, "R1": 0.5, "RL": 0.5},
    }

    comparison = _compare_revision_metrics(
        selected_skill="manage-rag-default",
        baseline=metrics,
        revised=metrics,
        prediction="Alpha",
        gold_answers=("Alpha",),
    )

    assert comparison["improved"] is False
    assert "did not measurably improve" in comparison["reason"]


def test_load_task_workflow_config_resolves_ordered_tasks(tmp_path) -> None:
    """验证任务流配置按声明顺序解析数据、自优化配置和请求覆盖。"""
    data_path = tmp_path / "task.jsonl"
    _write_task_data(data_path, dataset="hotpotqa", question_id="q1")
    self_config_path = tmp_path / "self-opt.yaml"
    self_config_path.write_text(
        "schema_version: 1\n"
        f"framework_config: {FRAMEWORK_CONFIG_PATH.as_posix()}\n"
        "workspace:\n"
        "  runs_root: runs\n"
        "  run_id: null\n"
        "optimizer:\n"
        "  provider: openai-compatible\n"
        "  model: qwen3-8b\n"
        "  base_url: http://127.0.0.1:8000/v1\n"
        "  api_key: null\n"
        "  api_key_env: null\n",
        encoding="utf-8",
    )
    task_config_path = tmp_path / "task-flow.yaml"
    task_config_path.write_text(
        "schema_version: 1\n"
        "self_optimization_config: self-opt.yaml\n"
        "workflow:\n"
        "  max_process_examples: 3\n"
        "  tasks:\n"
        "    - name: first-task\n"
        "      dataset: hotpotqa\n"
        "      data_path: task.jsonl\n"
        "      max_examples: 1\n"
        "      request:\n"
        "        top_k: 2\n",
        encoding="utf-8",
    )

    config = load_task_workflow_config(task_config_path)

    assert config.max_process_examples == 3
    assert [task.name for task in config.tasks] == ["first-task"]
    assert config.tasks[0].data_path == data_path.resolve()
    assert config.tasks[0].request == {"top_k": 2}
    assert config.self_optimization.optimizer.model == "qwen3-8b"


def test_task_workflow_optimizes_after_each_task_and_reuses_copy(tmp_path) -> None:
    """验证每个 task 评估后优化一次，且后续 task 读取上一轮 Skill 修改。"""
    first_data = tmp_path / "first.jsonl"
    second_data = tmp_path / "second.jsonl"
    _write_task_data(first_data, dataset="hotpotqa", question_id="q1")
    _write_task_data(second_data, dataset="triviaqa", question_id="q2")
    framework_config = load_framework_config(FRAMEWORK_CONFIG_PATH)
    framework_config = replace(
        framework_config,
        request_defaults={**framework_config.request_defaults, "top_k": 2},
    )
    self_config = SelfOptimizationConfig(
        config_path=tmp_path / "self-opt.yaml",
        framework=framework_config,
        optimizer=framework_config.executor,
        runs_root=tmp_path / "runs",
        run_id="two-tasks",
        optimization=OptimizationSettings(
            max_examples=None,
            max_skill_file_chars=50_000,
        ),
    )
    config = TaskWorkflowConfig(
        config_path=tmp_path / "task-flow.yaml",
        self_optimization=self_config,
        tasks=(
            RAGTaskConfig("first", "hotpotqa", first_data, max_examples=1),
            RAGTaskConfig("second", "triviaqa", second_data, max_examples=1),
        ),
        max_process_examples=1,
    )
    executor = ScriptedModel(_executor_responses() + _executor_responses())
    source_path = SKILL_ROOT / "manage" / "manage-rag-default" / "SKILL.md"
    source_content = source_path.read_text(encoding="utf-8")
    first_update = source_content + "\nTask-one optimization guidance.\n"
    second_update = first_update + "\nTask-two optimization guidance.\n"
    optimizer = ScriptedModel(
        [
            _selection_response("manage-rag-default", "first task feedback"),
            _invalid_ownership_response(),
            _optimization_response(first_update, "first task feedback"),
            _selection_response("manage-rag-default", "second task feedback"),
            _invalid_ownership_response(),
            _optimization_response(second_update, "second task feedback"),
        ]
    )

    report = run_task_workflow(
        config,
        executor_model=executor,
        optimizer_model=optimizer,
        generation_evaluator=FixedGenerationEvaluator(),
        verbose=False,
    )

    assert report["completed_tasks"] == 2
    assert report["overall_summary"]["count"] == 2
    assert [task["optimization"]["status"] for task in report["tasks"]] == [
        "accepted",
        "accepted",
    ]
    copied_path = (
        Path(report["working_skill_root"])
        / "manage"
        / "manage-rag-default"
        / "SKILL.md"
    )
    assert copied_path.read_text(encoding="utf-8") == second_update
    assert source_path.read_text(encoding="utf-8") == source_content
    assert "Task-one optimization guidance." in executor.calls[4][0]
    assert len(optimizer.calls) == 6
    assert '"task_name": "first"' in optimizer.calls[0][0]
    assert optimizer.calls[0][1] is not None
    assert "PROPOSAL ATTEMPT 1 REJECTED" in optimizer.calls[2][0]
    assert '"task_name": "second"' in optimizer.calls[3][0]
    assert "PROPOSAL ATTEMPT 1 REJECTED" in optimizer.calls[5][0]
    first_audit = report["tasks"][0]["optimization"]["audit"]
    second_audit = report["tasks"][1]["optimization"]["audit"]
    assert Path(first_audit["prompt_path"]).read_text(encoding="utf-8") == (
        optimizer.calls[2][0]
    )
    assert Path(second_audit["prompt_path"]).read_text(encoding="utf-8") == (
        optimizer.calls[5][0]
    )
    assert len(first_audit["attempts"]) == 3
    assert len(second_audit["attempts"]) == 3
    assert first_audit["attempts"][0]["stage"] == "selection"
    first_attempt = first_audit["attempts"][1]
    assert "does not exist in selected Skill" in first_attempt["error"]
    assert Path(first_attempt["artifacts"]["raw_response_path"]).is_file()
    assert Path(first_attempt["artifacts"]["proposal_path"]).is_file()
    assert Path(first_audit["raw_response_path"]).is_file()
    assert Path(first_audit["proposal_path"]).is_file()
    assert Path(first_audit["result_path"]).is_file()

    events_path = Path(report["artifacts"]["log_path"])
    event_names = [
        json.loads(line)["event"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    first_evaluation = event_names.index("task_evaluation_completed")
    first_optimization = event_names.index("task_optimization_requested")
    first_completed = event_names.index("task_completed")
    second_started = event_names.index("task_started", first_completed + 1)
    assert first_evaluation < first_optimization < first_completed < second_started
    assert event_names[-1] == "workflow_completed"


def test_task_workflow_continues_when_optimizer_has_no_change(tmp_path) -> None:
    """验证优化器连续照抄 Skill 时记录 skipped，并正常完成后续流程。"""
    data_path = tmp_path / "task.jsonl"
    _write_task_data(data_path, dataset="hotpotqa", question_id="q1")
    framework_config = load_framework_config(FRAMEWORK_CONFIG_PATH)
    self_config = SelfOptimizationConfig(
        config_path=tmp_path / "self-opt.yaml",
        framework=framework_config,
        optimizer=framework_config.executor,
        runs_root=tmp_path / "runs",
        run_id="skipped-task",
        optimization=OptimizationSettings(
            max_skill_file_chars=50_000,
            max_proposal_attempts=2,
        ),
    )
    config = TaskWorkflowConfig(
        config_path=tmp_path / "task-flow.yaml",
        self_optimization=self_config,
        tasks=(RAGTaskConfig("first", "hotpotqa", data_path, max_examples=1),),
        max_process_examples=1,
    )
    unchanged = (
        SKILL_ROOT
        / "components"
        / "component-grounded-generator"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    optimizer = ScriptedModel(
        [
            _selection_response(
                "component-grounded-generator",
                "The generation feedback suggests checking answer formatting.",
            ),
            json.dumps({"edits": [{"path": "SKILL.md", "content": unchanged}]}),
            json.dumps({"edits": [{"path": "SKILL.md", "content": unchanged}]}),
        ]
    )

    report = run_task_workflow(
        config,
        executor_model=ScriptedModel(_executor_responses()),
        optimizer_model=optimizer,
        generation_evaluator=FixedGenerationEvaluator(),
        verbose=False,
    )

    optimization = report["tasks"][0]["optimization"]
    assert report["completed_tasks"] == 1
    assert optimization["status"] == "skipped"
    assert optimization["selected_skill"] == "component-grounded-generator"
    assert "no substantive change" in optimization["reason"]
    assert not any(Path(report["artifacts"]["revisions_root"]).iterdir())


def _write_task_data(path: Path, *, dataset: str, question_id: str) -> None:
    """写入一个含答案、候选文档和证据标签的最小规范 task。"""
    record = {
        "id": question_id,
        "dataset": dataset,
        "question": "Which token is the answer?",
        "answers": ["Alpha"],
        "documents": [
            {"id": "gold", "title": "Answer", "text": "The answer token is Alpha."},
            {"id": "noise", "title": "Noise", "text": "Unrelated material."},
        ],
        "relevant_document_ids": ["gold"],
        "label_type": "document",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _executor_responses() -> list[str]:
    """返回一次 Sequential RAG 三级选择和生成所需的固定模型响应。"""
    return [
        json.dumps(
            {
                "agentic_selection_guidance": "Use a direct lexical route.",
                "reason": "The question is simple.",
            }
        ),
        json.dumps(
            {
                "selected_agentic_skill": "agentic-sequential-skill",
                "reason": "One pass is sufficient.",
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
                "reason": "Use exact lexical evidence.",
            }
        ),
        "Alpha",
    ]


def _optimization_response(content: str, rationale: str) -> str:
    """构造一次只修改 Manage Skill 的合法优化器 JSON 响应。"""
    del rationale
    return json.dumps(
        {
            "edits": [{"path": "SKILL.md", "content": content}],
        }
    )


def _invalid_ownership_response() -> str:
    """构造把 Component 文件错误归入 Agentic Skill 的首次提案。"""
    return json.dumps(
        {
            "edits": [
                {
                    "path": "scripts/component.py",
                    "content": "def run(inputs, context):\n    return {'answer': 'Alpha'}\n",
                }
            ],
        }
    )


def _selection_response(skill: str, rationale: str) -> str:
    """构造优化第一阶段只选择目标 Skill 的严格 JSON。"""
    return json.dumps({"selected_skill": skill, "rationale": rationale})
