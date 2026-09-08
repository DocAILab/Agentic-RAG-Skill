from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from framework import load_framework_config
from self_optimization import (
    OptimizationProposal,
    OptimizationNoChange,
    OptimizationSettings,
    RevisionRejected,
    SelfOptimizationConfig,
    SkillEdit,
    SkillOptimizer,
    SkillWorkspace,
    load_self_optimization_config,
    run_self_optimization,
)

PROJECT_ROOT = Path(__file__).parents[1]
SKILL_ROOT = PROJECT_ROOT / "framework" / "skills"
FRAMEWORK_CONFIG_PATH = PROJECT_ROOT / "framework" / "settings.example.yaml"


class ScriptedModel:
    """按顺序返回预设文本并保存全部模型调用。"""

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
    """返回固定生成指标，避免测试加载 PPL 模型。"""

    def evaluate(self, prediction, references):
        """为任意预测返回完整的 XRAG 生成指标映射。"""
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


def test_load_self_optimization_config_keeps_optimizer_independent(tmp_path) -> None:
    """验证优化模型、工作目录和基础 framework 配置分别解析。"""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        "schema_version: 1\n"
        f"framework_config: {FRAMEWORK_CONFIG_PATH.as_posix()}\n"
        "workspace:\n"
        "  runs_root: runs\n"
        "  run_id: trial-1\n"
        "optimizer:\n"
        "  provider: openai-compatible\n"
        "  model: Qwen/Qwen3-8B\n"
        "  base_url: http://127.0.0.1:8000/v1\n"
        "  api_key: null\n"
        "  api_key_env: null\n"
        "optimization:\n"
        "  max_examples: 2\n",
        encoding="utf-8",
    )

    config = load_self_optimization_config(config_path)

    assert config.optimizer.model == "Qwen/Qwen3-8B"
    assert config.framework.skill_root == SKILL_ROOT
    assert config.runs_root == tmp_path / "runs"
    assert config.run_id == "trial-1"
    assert config.optimization.max_examples == 2
    assert config.optimization.max_tokens == 8192


def test_workspace_applies_revision_only_to_skill_copy(tmp_path) -> None:
    """验证通过校验的修订只改变运行副本并生成快照和 diff。"""
    source_path = SKILL_ROOT / "manage" / "manage-rag-default" / "SKILL.md"
    source_before = source_path.read_text(encoding="utf-8")
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="copy-only")
    updated = source_before + "\nOptimization note: favor lower-cost valid plans.\n"

    revision = workspace.apply_proposal(
        OptimizationProposal(
            selected_skill="manage-rag-default",
            rationale="The previous route used unnecessary complexity.",
            edits=(SkillEdit(path="SKILL.md", content=updated),),
        ),
        candidate_names=("manage-rag-default", "agentic-sequential-skill"),
        max_edits=3,
        max_file_chars=50_000,
        allow_new_files=False,
    )

    copied_path = workspace.skill_root / "manage" / "manage-rag-default" / "SKILL.md"
    assert source_path.read_text(encoding="utf-8") == source_before
    assert copied_path.read_text(encoding="utf-8") == updated
    assert revision.diff_path.is_file()
    assert "Optimization note" in revision.diff_path.read_text(encoding="utf-8")
    assert (revision.revision_dir / "before" / "SKILL.md").is_file()


def test_workspace_rejects_protected_and_invalid_runtime_edits(tmp_path) -> None:
    """验证受保护清单不可修改，非法 Python 入口会自动恢复。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="rollback")
    component_dir = (
        workspace.skill_root
        / "components"
        / "component-bm25-retriever"
    )
    runtime_path = component_dir / "scripts" / "component.py"
    runtime_before = runtime_path.read_text(encoding="utf-8")

    with pytest.raises(RevisionRejected, match="Protected Skill file"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="component-bm25-retriever",
                rationale="Change the contract.",
                edits=(SkillEdit(path="ragskill.yaml", content="kind: manage\n"),),
            ),
            candidate_names=("component-bm25-retriever",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    with pytest.raises(RevisionRejected, match="rolled back"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="component-bm25-retriever",
                rationale="Break the runtime signature.",
                edits=(
                    SkillEdit(
                        path="scripts/component.py",
                        content="def run(wrong):\n    return {}\n",
                    ),
                ),
            ),
            candidate_names=("component-bm25-retriever",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    assert runtime_path.read_text(encoding="utf-8") == runtime_before
    assert (
        workspace.revisions_root / "revision-0001" / "result.json"
    ).is_file()


def test_workspace_rejects_terminal_newline_only_revision(tmp_path) -> None:
    """验证仅改变换行风格或文件末尾换行不会被记录成有效 Skill 优化。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="newline-only")
    copied_path = (
        workspace.skill_root / "manage" / "manage-rag-default" / "SKILL.md"
    )
    normalized_without_terminal_newline = copied_path.read_text(
        encoding="utf-8"
    ).rstrip("\n")

    with pytest.raises(RevisionRejected, match="no textual diff"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="manage-rag-default",
                rationale="Only normalize line endings.",
                edits=(
                    SkillEdit(
                        path="SKILL.md",
                        content=normalized_without_terminal_newline,
                    ),
                ),
            ),
            candidate_names=("manage-rag-default",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )


def test_workspace_rejects_undefined_python_name(tmp_path) -> None:
    """验证包含未定义名称的 Python 修改在写入副本后被静态检查并回滚。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="undefined-name")
    runtime_path = (
        workspace.skill_root
        / "agentic"
        / "agentic-sequential-skill"
        / "scripts"
        / "workflow.py"
    )
    original = runtime_path.read_text(encoding="utf-8")
    invalid = original.replace(
        '    original_query = str(request["query"])',
        '    _missing_helper()\n    original_query = str(request["query"])',
        1,
    )

    with pytest.raises(RevisionRejected, match="static validation"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="agentic-sequential-skill",
                rationale="Call an unavailable helper.",
                edits=(SkillEdit(path="scripts/workflow.py", content=invalid),),
            ),
            candidate_names=("agentic-sequential-skill",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    assert runtime_path.read_text(encoding="utf-8") == original


def test_workspace_rejects_undeclared_agentic_slot(tmp_path) -> None:
    """验证 Agentic 修改不能绕过 workflow 槽位调用不存在的具体组件名称。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="invalid-slot")
    runtime_path = (
        workspace.skill_root
        / "agentic"
        / "agentic-sequential-skill"
        / "scripts"
        / "workflow.py"
    )
    original = runtime_path.read_text(encoding="utf-8")
    invalid = original.replace(
        '        "retriever",',
        '        "bm25_retriever",',
        1,
    )

    with pytest.raises(RevisionRejected, match="undeclared component slots"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="agentic-sequential-skill",
                rationale="Call a concrete retriever name directly.",
                edits=(SkillEdit(path="scripts/workflow.py", content=invalid),),
            ),
            candidate_names=("agentic-sequential-skill",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    assert runtime_path.read_text(encoding="utf-8") == original


def test_workspace_rejects_component_output_contract_mismatch(tmp_path) -> None:
    """验证检索组件不能被替换为返回生成答案的其他组件实现。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="wrong-capability")
    runtime_path = (
        workspace.skill_root
        / "components"
        / "component-bm25-retriever"
        / "scripts"
        / "component.py"
    )
    original = runtime_path.read_text(encoding="utf-8")
    generator_source = (
        workspace.skill_root
        / "components"
        / "component-grounded-generator"
        / "scripts"
        / "component.py"
    ).read_text(encoding="utf-8")

    with pytest.raises(RevisionRejected, match="output contract"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="component-bm25-retriever",
                rationale="Incorrectly replace retrieval with generation.",
                edits=(
                    SkillEdit(
                        path="scripts/component.py",
                        content=generator_source,
                    ),
                ),
            ),
            candidate_names=("component-bm25-retriever",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    assert runtime_path.read_text(encoding="utf-8") == original


def test_workspace_rejects_unused_python_import(tmp_path) -> None:
    """验证只添加未使用 import 的 Python 提案不会成为有效 revision。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="unused-import")
    runtime_path = (
        workspace.skill_root
        / "agentic"
        / "agentic-sequential-skill"
        / "scripts"
        / "workflow.py"
    )
    original = runtime_path.read_text(encoding="utf-8")
    invalid = original.replace("\n\ndef run", "\n\nimport re\n\ndef run", 1)

    with pytest.raises(RevisionRejected, match="F401"):
        workspace.apply_proposal(
            OptimizationProposal(
                selected_skill="agentic-sequential-skill",
                rationale="Add an unused import.",
                edits=(SkillEdit(path="scripts/workflow.py", content=invalid),),
            ),
            candidate_names=("agentic-sequential-skill",),
            max_edits=3,
            max_file_chars=50_000,
            allow_new_files=False,
        )

    assert runtime_path.read_text(encoding="utf-8") == original


def test_optimizer_receives_metrics_trace_and_candidate_files(tmp_path) -> None:
    """验证优化提示同时包含评估、检索过程和本轮候选 Skill。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="prompt")
    skill_path = workspace.skill_root / "manage" / "manage-rag-default" / "SKILL.md"
    updated = skill_path.read_text(encoding="utf-8") + "\nPrefer direct evidence.\n"
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "selected_skill": "manage-rag-default",
                    "rationale": "Hit@1 is zero after a noisy retrieval route.",
                }
            ),
            json.dumps({"edits": [{"path": "SKILL.md", "content": updated}]}),
        ]
    )
    optimizer = SkillOptimizer(
        model,
        OptimizationSettings(max_examples=1, max_skill_file_chars=50_000),
    )

    proposal = optimizer.propose(
        workspace=workspace,
        candidate_names=("manage-rag-default", "agentic-sequential-skill"),
        evaluation={"metrics": {"retrieval": {"Hit@1": 0.0}}},
        retrieval_process={
            "trace": [{"step": "retrieve", "document_count": 10}],
            "selection": {"agentic_skill": "agentic-sequential-skill"},
        },
    )

    selection_prompt = model.calls[0][0]
    prompt = model.calls[1][0]
    assert proposal.selected_skill == "manage-rag-default"
    assert "CANDIDATE SKILL CATALOG" in selection_prompt
    assert "SELECTED SKILL SNAPSHOT" not in selection_prompt
    assert "does not make Generator or other\nSkill files frozen" in model.calls[0][1]
    assert "EVALUATION FEEDBACK" in prompt
    assert "selected Skill's files marked\neditable=true are still editable" in (
        model.calls[1][1]
    )
    assert '"Hit@1": 0.0' in prompt
    assert "RETRIEVAL PROCESS" in prompt
    assert '"step": "retrieve"' in prompt
    assert '"editable": false' in prompt
    assert "ragskill.yaml" in prompt


def test_optimizer_retries_python_docstring_only_change(tmp_path) -> None:
    """验证 Python 仅改 docstring 会被反馈，并在下一次尝试改为行为变化。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="ast-retry")
    workflow_path = (
        workspace.skill_root
        / "agentic"
        / "agentic-sequential-skill"
        / "scripts"
        / "workflow.py"
    )
    source = workflow_path.read_text(encoding="utf-8")
    docstring_only = source.replace(
        "只包含抽象组件调用的顺序式 RAG workflow。",
        "顺序式 RAG workflow，支持多文档。",
        1,
    )
    behavioral = source.replace(
        'request.get("top_k", 3)',
        'request.get("top_k", 4)',
        1,
    )
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "selected_skill": "agentic-sequential-skill",
                    "rationale": "Describe multi-document support.",
                }
            ),
            json.dumps(
                {
                    "edits": [
                        {"path": "scripts/workflow.py", "content": docstring_only}
                    ],
                }
            ),
            json.dumps(
                {
                    "edits": [
                        {"path": "scripts/workflow.py", "content": behavioral}
                    ],
                }
            ),
        ]
    )
    optimizer = SkillOptimizer(
        model,
        OptimizationSettings(
            max_examples=1,
            max_skill_file_chars=50_000,
            max_proposal_attempts=2,
        ),
    )

    proposal = optimizer.propose(
        workspace=workspace,
        candidate_names=("agentic-sequential-skill",),
        evaluation={"metrics": {"retrieval": {"Hit@1": 0.0}}},
        retrieval_process={"trace": [{"step": "retrieve"}]},
    )

    assert proposal.edits[0].content == behavioral
    assert len(optimizer.calls) == 3
    assert optimizer.calls[0].stage == "selection"
    assert "does not make a substantive change" in optimizer.calls[1].error
    assert "PROPOSAL ATTEMPT 1 REJECTED" in optimizer.calls[2].prompt


def test_optimizer_classifies_repeated_unchanged_edits_as_no_change(tmp_path) -> None:
    """验证连续照抄已选 Skill 时返回可区分的 no-change 结果。"""
    workspace = SkillWorkspace.create(SKILL_ROOT, tmp_path, run_id="no-change")
    skill_path = (
        workspace.skill_root
        / "components"
        / "component-grounded-generator"
        / "SKILL.md"
    )
    unchanged = skill_path.read_text(encoding="utf-8")
    model = ScriptedModel(
        [
            json.dumps(
                {
                    "selected_skill": "component-grounded-generator",
                    "rationale": "The answer formatting metrics are weak.",
                }
            ),
            json.dumps({"edits": [{"path": "SKILL.md", "content": unchanged}]}),
            json.dumps({"edits": [{"path": "SKILL.md", "content": unchanged}]}),
        ]
    )
    optimizer = SkillOptimizer(
        model,
        OptimizationSettings(
            max_skill_file_chars=50_000,
            max_proposal_attempts=2,
        ),
    )

    with pytest.raises(OptimizationNoChange, match="no substantive change") as error:
        optimizer.propose(
            workspace=workspace,
            candidate_names=("component-grounded-generator",),
            evaluation={"metrics": {"generation": {"WER": 1.0}}},
            retrieval_process={"prediction": "Yes."},
        )

    assert error.value.selected_skill == "component-grounded-generator"
    assert error.value.attempts == 2
    assert [call.stage for call in optimizer.calls] == [
        "selection",
        "editing",
        "editing",
    ]


def test_run_self_optimization_completes_one_feedback_revision(tmp_path) -> None:
    """验证一次查询形成执行、评估、Qwen 修改和副本提交的完整闭环。"""
    framework_config = load_framework_config(FRAMEWORK_CONFIG_PATH)
    assert framework_config.demo is not None
    framework_config = replace(
        framework_config,
        demo=replace(
            framework_config.demo,
            candidate_documents_only=True,
            max_examples=1,
        ),
    )
    config = SelfOptimizationConfig(
        config_path=tmp_path / "settings.yaml",
        framework=framework_config,
        optimizer=framework_config.executor,
        runs_root=tmp_path / "runs",
        run_id="end-to-end",
        optimization=OptimizationSettings(
            max_examples=1,
            max_skill_file_chars=50_000,
        ),
    )
    executor = ScriptedModel(
        [
            json.dumps(
                {
                    "agentic_selection_guidance": "Use one lexical route.",
                    "reason": "The query contains exact names.",
                }
            ),
            json.dumps(
                {
                    "selected_agentic_skill": "agentic-sequential-skill",
                    "reason": "One route is sufficient.",
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
                    "reason": "Use exact lexical anchors.",
                }
            ),
            "The Los Angeles Dance Theater",
        ]
    )
    source_manage_path = SKILL_ROOT / "manage" / "manage-rag-default" / "SKILL.md"
    source_manage = source_manage_path.read_text(encoding="utf-8")
    optimized_manage = source_manage + "\nUse retrieval failures as future guidance.\n"
    qwen_selection = json.dumps(
        {
            "selected_skill": "manage-rag-default",
            "rationale": "The evaluated route reveals reusable guidance.",
        }
    )
    qwen_response = json.dumps(
        {"edits": [{"path": "SKILL.md", "content": optimized_manage}]}
    )
    qwen = ScriptedModel([qwen_selection, qwen_response])

    report = run_self_optimization(
        config,
        executor_model=executor,
        optimizer_model=qwen,
        generation_evaluator=FixedGenerationEvaluator(),
        verbose=False,
    )

    assert report["summary"]["count"] == 1
    assert report["examples"][0]["optimization"]["status"] == "accepted"
    assert report["examples"][0]["optimization"]["selected_skill"] == (
        "manage-rag-default"
    )
    copied_manage = (
        Path(report["working_skill_root"])
        / "manage"
        / "manage-rag-default"
        / "SKILL.md"
    )
    assert copied_manage.read_text(encoding="utf-8") == optimized_manage
    assert source_manage_path.read_text(encoding="utf-8") == source_manage
    assert "CANDIDATE SKILL CATALOG" in qwen.calls[0][0]
    assert "EVALUATION FEEDBACK" in qwen.calls[1][0]
    assert "RETRIEVAL PROCESS" in qwen.calls[1][0]
    audit = report["examples"][0]["optimization"]["audit"]
    assert Path(audit["prompt_path"]).read_text(encoding="utf-8") == qwen.calls[1][0]
    assert Path(audit["raw_response_path"]).read_text(encoding="utf-8") == qwen_response
    proposal_log = json.loads(
        Path(audit["proposal_path"]).read_text(encoding="utf-8")
    )
    assert proposal_log["selected_skill"] == "manage-rag-default"
    assert Path(audit["result_path"]).is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "runs" / "end-to-end" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    names = [event["event"] for event in events]
    assert names.index("evaluation_completed") < names.index(
        "optimization_requested"
    )
    assert names[-1] == "run_completed"
