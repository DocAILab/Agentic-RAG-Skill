from __future__ import annotations

import ast
from pathlib import Path

import pytest

from framework import SkillKind, SkillSpecError, discover_specs, load_runtime_callable
from framework.spec import binding_requirement_errors

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
    def __init__(self, model_response="Apples grow in orchards."):
        """初始化用于断言模型调用内容的提示词记录。"""
        self.prompts = []
        self.model_response = model_response

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
        return self.model_response


def _specs_by_name():
    """发现全部样例 Skill，并按包名建立索引。"""
    return {spec.package_name: spec for spec in discover_specs(SAMPLE_ROOT)}


def test_sample_repository_has_three_strict_skill_levels() -> None:
    """验证样例仓库包含三层必需 Skill。"""
    specs = tuple(discover_specs(SAMPLE_ROOT))

    packages_by_kind = {
        kind: {
            spec.package_name
            for spec in specs
            if spec.kind is kind
        }
        for kind in SkillKind
    }

    assert set(packages_by_kind) == set(SkillKind)
    assert {"manage-rag-default"} <= packages_by_kind[SkillKind.MANAGE]
    assert {
        "agentic-conditional-rag",
        "agentic-hybrid-rag",
        "agentic-iterative-rag",
        "agentic-sequential-skill",
        "agentic-parallel-rag",
    } <= packages_by_kind[SkillKind.AGENTIC]
    assert {
        "component-bge-reranker",
        "component-bm25-retriever",
        "component-classifier",
        "component-critic",
        "component-grounded-generator",
        "component-hyde-rewriter",
        "component-vector-retriever",
    } <= packages_by_kind[SkillKind.COMPONENT]


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


def test_hyde_declares_vector_retriever_requirement() -> None:
    """验证 HyDE manifest 声明必须搭配 Vector Retriever。"""
    hyde = _specs_by_name()["component-hyde-rewriter"]

    assert len(hyde.requires) == 1
    assert hyde.requires[0].capability == "retriever"
    assert hyde.requires[0].components == ("component-vector-retriever",)


def test_hyde_requirement_accepts_vector_with_another_retriever() -> None:
    """验证存在 Vector 时额外绑定 BM25 不会破坏 HyDE 依赖。"""
    specs = _specs_by_name()
    conditional = specs["agentic-conditional-rag"]
    components = {
        spec.package_name: spec
        for spec in specs.values()
        if spec.kind is SkillKind.COMPONENT
    }
    bindings = {
        "classifier": ("component-classifier",),
        "rewriter": ("component-hyde-rewriter",),
        "lexical_retriever": ("component-bm25-retriever",),
        "semantic_retriever": ("component-vector-retriever",),
        "reranker": (),
        "generator": ("component-grounded-generator",),
    }

    assert binding_requirement_errors(
        conditional,
        bindings,
        components,
    ) == ()


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


def test_sequential_workflow_calls_retriever_then_generator() -> None:
    """验证 Sequential workflow 按检索后生成的顺序调用组件。"""
    workflow = load_runtime_callable(_specs_by_name()["agentic-sequential-skill"])
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


def test_sequential_workflow_uses_hyde_only_for_retrieval() -> None:
    """验证 HyDE 改写只进入检索器，后续组件仍使用原始问题。"""
    workflow = load_runtime_callable(_specs_by_name()["agentic-sequential-skill"])
    original_query = "Where do apples grow?"
    hypothetical_document = "Apple trees grow in temperate orchards."
    components = FakeComponents(
        {
            "rewriter": [
                lambda inputs: {"rewritten_query": hypothetical_document}
            ],
            "retriever": [
                lambda inputs: {"documents": [DOCUMENTS[0]]}
            ],
            "reranker": [
                lambda inputs: {"documents": list(inputs["documents"])}
            ],
            "generator": [lambda inputs: {"answer": "orchards"}],
        }
    )

    result = workflow(
        {
            "query": original_query,
            "documents": DOCUMENTS,
            "top_k": 1,
            "rewrite_temperature": 0.25,
            "rewrite_max_tokens": 96,
        },
        components,
    )

    calls = {slot: inputs for slot, _, inputs in components.calls}
    assert result["answer"] == "orchards"
    assert [call[0] for call in components.calls] == [
        "rewriter",
        "retriever",
        "reranker",
        "generator",
    ]
    assert calls["rewriter"]["query"] == original_query
    assert calls["rewriter"]["temperature"] == 0.25
    assert calls["rewriter"]["max_tokens"] == 96
    assert calls["retriever"]["query"] == hypothetical_document
    assert calls["reranker"]["query"] == original_query
    assert calls["generator"]["query"] == original_query


@pytest.mark.parametrize(
    "rewriter_result",
    [
        {},
        {"rewritten_query": None},
        {"rewritten_query": "   "},
    ],
)
def test_sequential_workflow_rejects_invalid_rewriter_result(
    rewriter_result,
) -> None:
    """验证 workflow 拒绝缺失、非字符串或空白的改写结果。"""
    workflow = load_runtime_callable(_specs_by_name()["agentic-sequential-skill"])
    components = FakeComponents(
        {
            "rewriter": [lambda inputs: rewriter_result],
            "retriever": [lambda inputs: {"documents": []}],
            "generator": [lambda inputs: {"answer": "unused"}],
        }
    )

    with pytest.raises(
        ValueError,
        match="Rewriter must return a non-empty rewritten_query",
    ):
        workflow({"query": "Where?"}, components)


def test_conditional_workflow_uses_only_lexical_route() -> None:
    """验证 lexical 路线只使用原问题调用词法检索器。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    original_query = "What is product code XR-100?"
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "lexical",
                    "reason": "The query contains an exact identifier.",
                    "confidence": 0.95,
                }
            ],
            "rewriter": [
                lambda inputs: pytest.fail(
                    "Lexical route must not call the Rewriter"
                )
            ],
            "lexical_retriever": [
                lambda inputs: {"documents": [DOCUMENTS[0]]}
            ],
            "semantic_retriever": [
                lambda inputs: pytest.fail(
                    "Lexical route must not call the semantic Retriever"
                )
            ],
            "generator": [
                lambda inputs: {"answer": "XR-100 is documented."}
            ],
        }
    )

    result = workflow(
        {
            "query": original_query,
            "documents": DOCUMENTS,
            "top_k": 1,
        },
        components,
    )

    calls = {
        slot: inputs
        for slot, _, inputs in components.calls
    }

    assert result["answer"] == "XR-100 is documented."
    assert result["route"] == "lexical"
    assert [call[0] for call in components.calls] == [
        "classifier",
        "lexical_retriever",
        "generator",
    ]
    assert calls["classifier"]["query"] == original_query
    assert calls["lexical_retriever"]["query"] == original_query
    assert calls["generator"]["query"] == original_query
    assert result["documents"] == [DOCUMENTS[0]]


def test_conditional_workflow_uses_hyde_only_for_semantic_retrieval() -> None:
    """验证 semantic 路线只把 HyDE 文本用于语义检索。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    original_query = "Where do apples usually grow?"
    hypothetical_document = (
        "Apple trees commonly grow in temperate orchards."
    )
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "semantic",
                    "reason": "The query is a semantic paraphrase.",
                    "confidence": 0.9,
                }
            ],
            "rewriter": [
                lambda inputs: {
                    "rewritten_query": hypothetical_document
                }
            ],
            "lexical_retriever": [
                lambda inputs: pytest.fail(
                    "Semantic route must not call the lexical Retriever"
                )
            ],
            "semantic_retriever": [
                lambda inputs: {"documents": [DOCUMENTS[0]]}
            ],
            "reranker": [
                lambda inputs: {
                    "documents": list(inputs["documents"])
                }
            ],
            "generator": [
                lambda inputs: {"answer": "They grow in orchards."}
            ],
        }
    )

    result = workflow(
        {
            "query": original_query,
            "documents": DOCUMENTS,
            "top_k": 1,
            "rewrite_temperature": 0.2,
            "rewrite_max_tokens": 96,
        },
        components,
    )

    calls = {
        slot: inputs
        for slot, _, inputs in components.calls
    }

    assert result["answer"] == "They grow in orchards."
    assert result["route"] == "semantic"
    assert [call[0] for call in components.calls] == [
        "classifier",
        "rewriter",
        "semantic_retriever",
        "reranker",
        "generator",
    ]
    assert calls["classifier"]["query"] == original_query
    assert calls["rewriter"]["query"] == original_query
    assert calls["rewriter"]["temperature"] == 0.2
    assert calls["rewriter"]["max_tokens"] == 96
    assert calls["semantic_retriever"]["query"] == hypothetical_document
    assert calls["reranker"]["query"] == original_query
    assert calls["generator"]["query"] == original_query
    assert calls["generator"]["documents"] == [DOCUMENTS[0]]
    assert result["documents"] == [DOCUMENTS[0]]


def test_conditional_workflow_fuses_hybrid_retrieval_routes() -> None:
    """验证 hybrid 路线使用不同查询检索并通过 RRF 融合。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    original_query = "Which fruit grows in an orchard?"
    hypothetical_document = (
        "Orchard fruit grows on cultivated trees."
    )
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "hybrid",
                    "reason": "Exact and semantic evidence are useful.",
                    "confidence": 0.85,
                }
            ],
            "rewriter": [
                lambda inputs: {
                    "rewritten_query": hypothetical_document
                }
            ],
            "lexical_retriever": [
                lambda inputs: {
                    "documents": [DOCUMENTS[0], DOCUMENTS[1]]
                }
            ],
            "semantic_retriever": [
                lambda inputs: {
                    "documents": [DOCUMENTS[1], DOCUMENTS[2]]
                }
            ],
            "generator": [
                lambda inputs: {"answer": "Apple is an orchard fruit."}
            ],
        }
    )

    result = workflow(
        {
            "query": original_query,
            "documents": DOCUMENTS,
            "top_k": 2,
            "rank_constant": 60,
        },
        components,
    )

    calls = {
        slot: inputs
        for slot, _, inputs in components.calls
    }

    assert result["answer"] == "Apple is an orchard fruit."
    assert result["route"] == "hybrid"
    assert [call[0] for call in components.calls] == [
        "classifier",
        "rewriter",
        "lexical_retriever",
        "semantic_retriever",
        "generator",
    ]
    assert calls["lexical_retriever"]["query"] == original_query
    assert calls["semantic_retriever"]["query"] == hypothetical_document
    assert calls["lexical_retriever"]["top_k"] == 4
    assert calls["semantic_retriever"]["top_k"] == 4
    assert calls["generator"]["query"] == original_query
    assert [
        document["id"]
        for document in calls["generator"]["documents"]
    ] == ["banana", "apple"]
    assert [document["id"] for document in result["documents"]] == [
        "banana",
        "apple",
    ]


def test_conditional_semantic_route_works_without_rewriter() -> None:
    """验证未绑定 Rewriter 时 semantic 路线使用原问题。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    original_query = "Where do apples grow?"
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "semantic",
                    "reason": "Semantic retrieval is appropriate.",
                    "confidence": 0.8,
                }
            ],
            "semantic_retriever": [
                lambda inputs: {"documents": [DOCUMENTS[0]]}
            ],
            "generator": [lambda inputs: {"answer": "In orchards."}],
        }
    )

    result = workflow(
        {"query": original_query, "documents": DOCUMENTS},
        components,
    )

    calls = {
        slot: inputs
        for slot, _, inputs in components.calls
    }
    assert [call[0] for call in components.calls] == [
        "classifier",
        "semantic_retriever",
        "generator",
    ]
    assert calls["semantic_retriever"]["query"] == original_query
    assert result["route"] == "semantic"


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_conditional_workflow_rejects_empty_query(query) -> None:
    """验证 Conditional workflow 拒绝空问题。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )

    with pytest.raises(ValueError, match="non-empty query"):
        workflow({"query": query}, FakeComponents({}))


@pytest.mark.parametrize("top_k", [0, -1])
def test_conditional_workflow_rejects_non_positive_top_k(top_k) -> None:
    """验证 Conditional workflow 要求 top_k 为正数。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        workflow(
            {"query": "Where?", "top_k": top_k},
            FakeComponents({}),
        )


@pytest.mark.parametrize("route", [None, "", "unsupported"])
def test_conditional_workflow_rejects_invalid_classifier_route(route) -> None:
    """验证 Conditional workflow 拒绝分类器返回未知路线。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": route,
                    "reason": "invalid test route",
                    "confidence": 0.5,
                }
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Classifier route must be lexical, semantic, or hybrid",
    ):
        workflow({"query": "Where?"}, components)


@pytest.mark.parametrize(
    "rewriter_result",
    [
        {},
        {"rewritten_query": None},
        {"rewritten_query": "   "},
    ],
)
def test_conditional_workflow_rejects_invalid_rewriter_result(
    rewriter_result,
) -> None:
    """验证 semantic 路线拒绝缺失或为空的改写结果。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "semantic",
                    "reason": "Semantic retrieval is appropriate.",
                    "confidence": 0.8,
                }
            ],
            "rewriter": [lambda inputs: rewriter_result],
        }
    )

    with pytest.raises(
        ValueError,
        match="Rewriter must return a non-empty rewritten_query",
    ):
        workflow({"query": "Where?"}, components)


@pytest.mark.parametrize("rank_constant", [0, -1])
def test_conditional_hybrid_route_rejects_invalid_rank_constant(
    rank_constant,
) -> None:
    """验证 hybrid 路线要求 RRF 融合常数为正数。"""
    workflow = load_runtime_callable(
        _specs_by_name()["agentic-conditional-rag"]
    )
    components = FakeComponents(
        {
            "classifier": [
                lambda inputs: {
                    "route": "hybrid",
                    "reason": "Both retrieval routes are useful.",
                    "confidence": 0.8,
                }
            ],
            "lexical_retriever": [
                lambda inputs: {"documents": []}
            ],
            "semantic_retriever": [
                lambda inputs: {"documents": []}
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="rank_constant must be positive",
    ):
        workflow(
            {
                "query": "Where?",
                "rank_constant": rank_constant,
            },
            components,
        )


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
    """验证 Component 样例提供可直接执行的具体实现。"""
    specs = _specs_by_name()
    context = FakeContext()

    bm25 = load_runtime_callable(specs["component-bm25-retriever"])
    vector = load_runtime_callable(specs["component-vector-retriever"])
    generator = load_runtime_callable(specs["component-grounded-generator"])
    hyde = load_runtime_callable(specs["component-hyde-rewriter"])

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
    assert hyde(
        {"query": "Where do apples grow?", "temperature": 0.0, "max_tokens": 64},
        context,
    )["rewritten_query"] == "Apples grow in orchards."
    assert "Where do apples grow?" in context.prompts[1][0]


def test_hyde_requires_query_field() -> None:
    """验证 HyDE 的 RewriteRequest 必须包含 query。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])

    with pytest.raises(KeyError, match="query"):
        hyde({}, FakeContext())


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_hyde_rejects_empty_query(query) -> None:
    """验证 HyDE 拒绝空字符串和仅含空白的查询。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])

    with pytest.raises(ValueError, match="non-empty query"):
        hyde({"query": query}, FakeContext())


@pytest.mark.parametrize("temperature", [-0.1, -1])
def test_hyde_rejects_negative_temperature(temperature) -> None:
    """验证 HyDE 的生成温度不能为负数。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        hyde(
            {"query": "Where?", "temperature": temperature},
            FakeContext(),
        )


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_hyde_rejects_non_positive_max_tokens(max_tokens) -> None:
    """验证 HyDE 的最大生成长度必须为正数。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        hyde(
            {"query": "Where?", "max_tokens": max_tokens},
            FakeContext(),
        )


@pytest.mark.parametrize(
    "optional_inputs",
    [
        {},
        {"temperature": None, "max_tokens": None},
    ],
)
def test_hyde_uses_defaults_and_calls_model_once(optional_inputs) -> None:
    """验证缺失或为 null 的可选参数使用默认值且只调用一次模型。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])
    context = FakeContext()

    result = hyde(
        {"query": "  Where do apples grow?  ", **optional_inputs},
        context,
    )

    assert result == {"rewritten_query": "Apples grow in orchards."}
    assert len(context.prompts) == 1
    prompt, temperature, max_tokens = context.prompts[0]
    assert "Question: Where do apples grow?" in prompt
    assert temperature == 0.0
    assert max_tokens == 8192


def test_hyde_accepts_minimum_positive_max_tokens() -> None:
    """验证 max_tokens=1 是允许的最小正整数边界。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])
    context = FakeContext()

    hyde(
        {"query": "Where?", "temperature": 0.0, "max_tokens": 1},
        context,
    )

    assert context.prompts[0][1:] == (0.0, 1)


def test_hyde_rejects_empty_model_output() -> None:
    """验证模型生成空白内容时 HyDE 不返回无效 RewriteResult。"""
    hyde = load_runtime_callable(_specs_by_name()["component-hyde-rewriter"])

    with pytest.raises(ValueError, match="empty hypothetical document"):
        hyde({"query": "Where?"}, FakeContext("  \n"))
