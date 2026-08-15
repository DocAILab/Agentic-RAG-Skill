from __future__ import annotations

import ast
from pathlib import Path

import pytest

from framework import SkillKind, SkillSpecError, discover_specs, load_runtime_callable

SAMPLE_ROOT = Path(__file__).parents[1] / "framework" / "skills"

DOCUMENTS = (
    {"id": "apple", "text": "Apple trees grow fruit in orchards."},
    {"id": "banana", "text": "Bananas are long yellow fruit."},
    {"id": "citrus", "text": "Lemons and oranges are citrus fruit."},
)


class FakeComponents:
    def __init__(self, bindings):
        """保存槽位绑定，并初始化组件调用记录。"""
        self.bindings = bindings
        self.calls = []

    def has(self, slot):
        """判断测试槽位是否包含可调用组件。"""
        return bool(self.bindings.get(slot))

    def call(self, slot, inputs, *, index=0):
        """记录并调用槽位中指定序号的测试组件。"""
        self.calls.append((slot, index, dict(inputs)))
        return self.bindings[slot][index](inputs)

    def call_all(self, slot, inputs):
        """依次调用测试槽位中的全部组件。"""
        return [
            self.call(slot, inputs, index=index)
            for index in range(len(self.bindings[slot]))
        ]


class FakeContext:
    def __init__(self):
        """初始化用于断言模型调用内容的提示词记录。"""
        self.prompts = []

    def embed(self, texts):
        """根据关键词生成确定性测试向量。"""
        return [
            (
                float("apple" in text.lower()),
                float("banana" in text.lower()),
                float("citrus" in text.lower()),
            )
            for text in texts
        ]

    def call_model(self, prompt, *, temperature=0.0, max_tokens=None):
        """记录生成参数并返回固定测试答案。"""
        self.prompts.append((prompt, temperature, max_tokens))
        return "Apples grow in orchards."


def _specs_by_name():
    """发现全部样例 Skill，并按包名建立索引。"""
    return {spec.package_name: spec for spec in discover_specs(SAMPLE_ROOT)}


def test_sample_repository_has_three_strict_skill_levels() -> None:
    """验证样例仓库严格包含三层共七个 Skill。"""
    specs = tuple(discover_specs(SAMPLE_ROOT))

    assert len(specs) == 7
    assert sum(spec.kind is SkillKind.MANAGE for spec in specs) == 1
    assert sum(spec.kind is SkillKind.AGENTIC for spec in specs) == 2
    assert sum(spec.kind is SkillKind.COMPONENT for spec in specs) == 4


def test_skill_packages_are_grouped_by_declared_kind() -> None:
    """验证每个 Skill 包都位于与清单 kind 对应的类型目录。"""
    expected_directories = {
        SkillKind.MANAGE: "manage",
        SkillKind.AGENTIC: "agentic",
        SkillKind.COMPONENT: "components",
    }

    for spec in discover_specs(SAMPLE_ROOT):
        assert spec.package_path.parent.name == expected_directories[spec.kind]


def test_discovery_rejects_skill_in_wrong_kind_directory(tmp_path) -> None:
    """验证发现器会拒绝目录层级与清单 kind 不一致的 Skill。"""
    package = tmp_path / "agentic" / "misplaced-manage"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\n"
        "name: misplaced-manage\n"
        "description: A deliberately misplaced Manage Skill for validation.\n"
        "---\n",
        encoding="utf-8",
    )
    (package / "ragskill.yaml").write_text(
        "schema_version: 1\n"
        "runtime_id: manage.test.misplaced\n"
        "kind: manage\n"
        "selection:\n"
        "  target_kind: agentic\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillSpecError, match="stored under 'agentic'"):
        discover_specs(tmp_path, validate_runtime=False)


def test_required_agentic_slots_have_compatible_component_samples() -> None:
    """验证每个必需 Agentic 槽位都有足量的接口兼容组件。"""
    specs = tuple(discover_specs(SAMPLE_ROOT))
    components = [spec for spec in specs if spec.kind is SkillKind.COMPONENT]

    for agentic in (spec for spec in specs if spec.kind is SkillKind.AGENTIC):
        for slot in agentic.slots:
            compatible = [
                capability
                for component in components
                for capability in component.provides
                if capability.name == slot.capability
                and capability.input_type == slot.input_type
                and capability.output_type == slot.output_type
            ]
            assert len(compatible) >= slot.min_count


def test_agentic_scripts_only_arrange_abstract_component_calls() -> None:
    """验证 Agentic 脚本不导入任何具体 Component 实现。"""
    for spec in discover_specs(SAMPLE_ROOT):
        if spec.kind is not SkillKind.AGENTIC:
            continue
        runtime_path = spec.package_path / spec.runtime.path
        tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert imports == []


def test_vanilla_workflow_calls_retriever_then_generator() -> None:
    """验证 Vanilla workflow 按检索后生成的顺序调用组件。"""
    workflow = load_runtime_callable(_specs_by_name()["agentic-vanilla-rag"])
    components = FakeComponents(
        {
            "retriever": [
                lambda inputs: {"documents": [DOCUMENTS[0]]}
            ],
            "generator": [lambda inputs: {"answer": "orchards"}],
        }
    )

    result = workflow(
        {"query": "Where do apples grow?", "documents": DOCUMENTS, "top_k": 1},
        components,
    )

    assert result["answer"] == "orchards"
    assert [call[0] for call in components.calls] == ["retriever", "generator"]


def test_rrfusion_workflow_calls_two_retrievers_and_fuses_results() -> None:
    """验证 RRFusion 调用两路检索器并正确融合重复结果。"""
    workflow = load_runtime_callable(_specs_by_name()["agentic-rrfusion"])
    components = FakeComponents(
        {
            "retrievers": [
                lambda inputs: {"documents": [DOCUMENTS[0], DOCUMENTS[1]]},
                lambda inputs: {"documents": [DOCUMENTS[0], DOCUMENTS[2]]},
            ],
            "generator": [lambda inputs: {"answer": "fused"}],
        }
    )

    result = workflow(
        {"query": "fruit", "documents": DOCUMENTS, "top_k": 2},
        components,
    )

    assert result["answer"] == "fused"
    assert result["documents"][0]["id"] == "apple"
    assert [call[0] for call in components.calls] == [
        "retrievers",
        "retrievers",
        "generator",
    ]


def test_component_samples_have_concrete_executable_implementations() -> None:
    """验证三个既有 Component 样例仍提供可直接执行的具体实现。"""
    specs = _specs_by_name()
    context = FakeContext()

    bm25 = load_runtime_callable(specs["component-bm25-retriever"])
    vector = load_runtime_callable(specs["component-vector-retriever"])
    generator = load_runtime_callable(specs["component-grounded-generator"])

    assert bm25(
        {"query": "yellow banana", "documents": DOCUMENTS, "top_k": 1},
        context,
    )["documents"][0]["id"] == "banana"
    assert vector(
        {"query": "apple", "documents": DOCUMENTS, "top_k": 1},
        context,
    )["documents"][0]["id"] == "apple"
    assert generator(
        {"query": "Where?", "documents": [DOCUMENTS[0]], "max_tokens": 64},
        context,
    )["answer"] == "Apples grow in orchards."
    assert "Apple trees grow" in context.prompts[0][0]
