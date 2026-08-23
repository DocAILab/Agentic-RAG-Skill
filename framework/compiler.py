"""将 Agentic workflow 与具体 Components 绑定并执行完整 RAG。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from .interfaces import ComponentContext
from .models import EmbeddingClient, ModelClient
from .selection import RAGSelectionPlan, select_rag_plan
from .spec import (
    RAGSkillSpec,
    SkillKind,
    SkillSpecError,
    SlotSpec,
    binding_requirement_errors,
    discover_specs,
    load_runtime_callable,
    load_spec,
)
from .vector_index import (
    DenseVectorIndex,
    build_or_load_vector_index,
    embedding_model_fingerprint,
    vector_index_cache_key,
)

ComponentCallable = Callable[[Mapping[str, Any], ComponentContext], Mapping[str, Any]]
WorkflowCallable = Callable[[Mapping[str, Any], Any], Mapping[str, Any]]


class CompilationError(ValueError):
    """表示选择计划无法安全绑定为可执行 RAG 命令。"""


class ExecutionError(RuntimeError):
    """表示已编译 workflow 或具体 Component 的运行结果不合法。"""


@dataclass(slots=True)
class RuntimeComponentContext:
    """把统一模型与向量接口注入具体 Components Skill。"""

    executor_model: ModelClient
    embedding_model: EmbeddingClient | None = None
    generation_system: str = (
        "You are the frozen Executor Model. Follow the Component prompt and do not "
        "modify model parameters or Skill definitions."
    )
    vector_index_cache_dir: Path | None = None
    embedding_fingerprint: str | None = None
    _embedding_cache: dict[str, tuple[float, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _embedding_cache_hits: int = field(default=0, init=False, repr=False)
    _embedding_cache_misses: int = field(default=0, init=False, repr=False)
    _vector_indexes: dict[str, DenseVectorIndex] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _vector_index_builds: int = field(default=0, init=False, repr=False)
    _vector_index_disk_loads: int = field(default=0, init=False, repr=False)
    _vector_index_memory_hits: int = field(default=0, init=False, repr=False)
    _vector_index_queries: int = field(default=0, init=False, repr=False)
    _last_vector_index: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """规范化缓存目录，并在未显式提供时推导 Embedding 模型指纹。"""
        if self.vector_index_cache_dir is not None:
            self.vector_index_cache_dir = Path(self.vector_index_cache_dir).resolve()
        if self.embedding_model is not None and self.embedding_fingerprint is None:
            self.embedding_fingerprint = embedding_model_fingerprint(self.embedding_model)

    def call_model(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """通过统一 ModelClient 调用冻结 Executor Model。"""
        return self.executor_model.generate(
            prompt,
            system=self.generation_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """只编码缓存中缺失的文本，并按原输入顺序返回全部向量。"""
        if self.embedding_model is None:
            raise ExecutionError(
                "Selected Component requires embeddings, but no embedding_model was provided"
            )
        normalized = [str(text) for text in texts]
        if not normalized:
            return []

        missing = []
        pending: set[str] = set()
        for text in normalized:
            if text not in self._embedding_cache and text not in pending:
                missing.append(text)
                pending.add(text)

        if missing:
            encoded = list(self.embedding_model.embed(missing))
            if len(encoded) != len(missing):
                raise ExecutionError(
                    "Embedding model vector count does not match uncached texts"
                )
            try:
                cached_vectors = [
                    tuple(float(value) for value in vector) for vector in encoded
                ]
            except (TypeError, ValueError) as exc:
                raise ExecutionError(
                    "Embedding model returned a non-numeric vector"
                ) from exc
            self._embedding_cache.update(
                dict(zip(missing, cached_vectors, strict=True))
            )

        self._embedding_cache_misses += len(missing)
        self._embedding_cache_hits += len(normalized) - len(missing)
        return [self._embedding_cache[text] for text in normalized]

    def embedding_cache_info(self) -> dict[str, int]:
        """返回当前运行中向量缓存的条目数、命中数和未命中数。"""
        return {
            "entries": len(self._embedding_cache),
            "hits": self._embedding_cache_hits,
            "misses": self._embedding_cache_misses,
        }

    def search_vector_index(
        self,
        *,
        query_text: str,
        document_ids: Sequence[str],
        document_texts: Sequence[str],
        top_k: int,
        text_format_version: str,
    ) -> Sequence[tuple[int, float]]:
        """加载或构建共享语料索引，并仅编码查询后执行向量检索。"""
        if self.embedding_model is None or self.embedding_fingerprint is None:
            raise ExecutionError(
                "Selected Component requires embeddings, but no embedding_model was provided"
            )
        normalized_ids = tuple(str(document_id) for document_id in document_ids)
        normalized_texts = tuple(str(text) for text in document_texts)
        cache_key = vector_index_cache_key(
            normalized_ids,
            normalized_texts,
            embedding_fingerprint=self.embedding_fingerprint,
            text_format_version=text_format_version,
        )
        index = self._vector_indexes.get(cache_key)
        if index is None:
            index = build_or_load_vector_index(
                normalized_ids,
                normalized_texts,
                embed=self.embedding_model.embed,
                embedding_fingerprint=self.embedding_fingerprint,
                text_format_version=text_format_version,
                cache_root=self.vector_index_cache_dir,
            )
            self._vector_indexes[cache_key] = index
            if index.source == "disk":
                self._vector_index_disk_loads += 1
            else:
                self._vector_index_builds += 1
            source = index.source
        else:
            self._vector_index_memory_hits += 1
            source = "memory"

        query_vectors = self.embed([query_text])
        if len(query_vectors) != 1:
            raise ExecutionError("Embedding model did not return one query vector")
        results = index.search(query_vectors[0], top_k)
        self._vector_index_queries += 1
        self._last_vector_index = {
            "cache_key": index.cache_key,
            "source": source,
            "cache_path": str(index.cache_path) if index.cache_path is not None else None,
            "document_count": len(index.document_ids),
            "dimension": index.dimension,
        }
        return results

    def vector_index_cache_info(self) -> dict[str, Any]:
        """返回持久化索引的构建、加载、内存命中及最近查询状态。"""
        return {
            "enabled": self.vector_index_cache_dir is not None,
            "cache_dir": (
                str(self.vector_index_cache_dir)
                if self.vector_index_cache_dir is not None
                else None
            ),
            "builds": self._vector_index_builds,
            "disk_loads": self._vector_index_disk_loads,
            "memory_hits": self._vector_index_memory_hits,
            "queries": self._vector_index_queries,
            "active_indexes": len(self._vector_indexes),
            "last_index": (
                dict(self._last_vector_index)
                if self._last_vector_index is not None
                else None
            ),
        }


@dataclass(slots=True)
class BoundComponentInvoker:
    """将 Agentic 槽位调用转发给已选中的具体 Component 函数。"""

    bindings: Mapping[str, tuple[ComponentCallable, ...]]
    binding_names: Mapping[str, tuple[str, ...]]
    context: ComponentContext
    _call_timings: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def start_run(self) -> None:
        """在执行新问题前清空上一题的组件耗时记录。"""
        self._call_timings.clear()

    def call_timings(self) -> list[dict[str, Any]]:
        """复制并返回当前问题已经完成的组件耗时记录。"""
        return [dict(timing) for timing in self._call_timings]

    def has(self, slot: str) -> bool:
        """判断槽位是否至少绑定了一个具体组件。"""
        return bool(self.bindings.get(slot))

    def call(
        self,
        slot: str,
        inputs: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> Mapping[str, Any]:
        """调用槽位中指定序号的组件，并校验其结构化返回值。"""
        functions = self.bindings.get(slot)
        if functions is None:
            raise ExecutionError(f"Workflow requested unknown Component slot: {slot}")
        if not 0 <= index < len(functions):
            raise ExecutionError(
                f"Component slot '{slot}' has no binding at index {index}"
            )
        component_name = self.binding_names[slot][index]
        started_at = perf_counter()
        try:
            result = functions[index](dict(inputs), self.context)
            if not isinstance(result, Mapping):
                raise ExecutionError(
                    f"Component slot '{slot}' returned a non-mapping result"
                )
        except Exception:
            self._call_timings.append(
                {
                    "slot": slot,
                    "component": component_name,
                    "duration_seconds": round(perf_counter() - started_at, 6),
                    "status": "failed",
                }
            )
            raise
        self._call_timings.append(
            {
                "slot": slot,
                "component": component_name,
                "duration_seconds": round(perf_counter() - started_at, 6),
                "status": "completed",
            }
        )
        return dict(result)

    def call_all(
        self,
        slot: str,
        inputs: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """按绑定顺序调用槽位中的全部组件。"""
        functions = self.bindings.get(slot)
        if functions is None:
            raise ExecutionError(f"Workflow requested unknown Component slot: {slot}")
        return [
            self.call(slot, inputs, index=index) for index in range(len(functions))
        ]


@dataclass(slots=True)
class CompiledRAGCommand:
    """保存已加载 workflow、组件绑定和可复现 Python 指令。"""

    workflow_name: str
    binding_names: Mapping[str, tuple[str, ...]]
    _workflow: WorkflowCallable = field(repr=False)
    _components: BoundComponentInvoker = field(repr=False)

    @property
    def instruction(self) -> str:
        """返回与当前编译结果等价的一行 Python 调用表达式。"""
        bindings = {
            slot: list(names) for slot, names in self.binding_names.items()
        }
        return (
            "run_compiled_rag("
            f"workflow={json.dumps(self.workflow_name)}, "
            f"bindings={json.dumps(bindings, ensure_ascii=False)}, "
            "request=request, skill_root=skill_root, context=context)"
        )

    def run(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """执行已绑定的 Agentic workflow，并校验最终结果。"""
        self._components.start_run()
        result = self._workflow(dict(request), self._components)
        if not isinstance(result, Mapping):
            raise ExecutionError("Agentic workflow returned a non-mapping result")
        normalized = dict(result)
        normalized["component_timings"] = self._components.call_timings()
        return normalized

    def __call__(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """允许将已编译命令作为普通 Python 函数直接调用。"""
        return self.run(request)


def compile_rag_command(
    plan: RAGSelectionPlan,
    *,
    skill_root: str | Path,
    context: ComponentContext,
) -> CompiledRAGCommand:
    """把经过校验的三级选择计划编译成可重复调用的 RAG 命令。"""
    return _compile_selected(
        workflow_name=plan.agentic_skill,
        binding_names=plan.component_bindings,
        skill_root=skill_root,
        context=context,
    )


def run_compiled_rag(
    *,
    workflow: str,
    bindings: Mapping[str, Sequence[str]],
    request: Mapping[str, Any],
    skill_root: str | Path,
    context: ComponentContext,
) -> dict[str, Any]:
    """通过显式 workflow 和绑定执行一条可复现的 Python RAG 指令。"""
    command = _compile_selected(
        workflow_name=workflow,
        binding_names=bindings,
        skill_root=skill_root,
        context=context,
    )
    return command.run(request)


def run_rag(
    request: Mapping[str, Any],
    *,
    model: ModelClient,
    skill_root: str | Path,
    embedding_model: EmbeddingClient | None = None,
    manage_skill: str = "manage-rag-default",
    vector_index_cache_dir: str | Path | None = None,
    embedding_fingerprint: str | None = None,
) -> dict[str, Any]:
    """用一次调用完成三级 LLM 选择、编译绑定、检索与答案生成。"""
    plan = select_rag_plan(
        request,
        model=model,
        skill_root=skill_root,
        manage_skill=manage_skill,
    )
    context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=embedding_model,
        vector_index_cache_dir=(
            Path(vector_index_cache_dir)
            if vector_index_cache_dir is not None
            else None
        ),
        embedding_fingerprint=embedding_fingerprint,
    )
    command = compile_rag_command(plan, skill_root=skill_root, context=context)
    result = command.run(request)
    result["selection"] = plan.to_dict()
    result["compiled_instruction"] = command.instruction
    return result


def _compile_selected(
    *,
    workflow_name: str,
    binding_names: Mapping[str, Sequence[str]],
    skill_root: str | Path,
    context: ComponentContext,
) -> CompiledRAGCommand:
    """解析选中包、复核槽位契约并构造运行时调用器。"""
    metadata = discover_specs(skill_root, validate_runtime=False)
    workflow_metadata = _find_named_spec(
        metadata,
        workflow_name,
        expected_kind=SkillKind.AGENTIC,
    )
    try:
        workflow_spec = load_spec(workflow_metadata.package_path)
        workflow = load_runtime_callable(workflow_spec)
    except SkillSpecError as exc:
        raise CompilationError(f"Cannot load Agentic workflow: {exc}") from exc

    expected_slots = {slot.name for slot in workflow_spec.slots}
    if set(binding_names) != expected_slots:
        raise CompilationError(
            f"Bindings must contain exactly these slots: {sorted(expected_slots)}"
        )

    component_metadata = {
        spec.package_name: spec for spec in metadata if spec.kind is SkillKind.COMPONENT
    }
    callable_bindings: dict[str, tuple[ComponentCallable, ...]] = {}
    normalized_bindings: dict[str, tuple[str, ...]] = {}
    for slot in workflow_spec.slots:
        names = _validate_binding_names(slot, binding_names[slot.name])
        functions = []
        for name in names:
            component = component_metadata.get(name)
            if component is None:
                raise CompilationError(f"Unknown Component Skill: {name}")
            if not _component_matches_slot(component, slot):
                raise CompilationError(
                    f"Component '{name}' is incompatible with slot '{slot.name}'"
                )
            try:
                validated = load_spec(component.package_path)
                function = load_runtime_callable(validated)
            except SkillSpecError as exc:
                raise CompilationError(f"Cannot load Component '{name}': {exc}") from exc
            functions.append(function)
        normalized_bindings[slot.name] = names
        callable_bindings[slot.name] = tuple(functions)

    requirement_errors = binding_requirement_errors(
        workflow_spec,
        normalized_bindings,
        component_metadata,
    )
    if requirement_errors:
        raise CompilationError(requirement_errors[0])

    return CompiledRAGCommand(
        workflow_name=workflow_spec.package_name,
        binding_names=normalized_bindings,
        _workflow=workflow,
        _components=BoundComponentInvoker(
            callable_bindings,
            normalized_bindings,
            context,
        ),
    )


def _find_named_spec(
    specs: Sequence[RAGSkillSpec],
    name: str,
    *,
    expected_kind: SkillKind,
) -> RAGSkillSpec:
    """查找指定层级的 Skill 元数据，并拒绝未知名称。"""
    for spec in specs:
        if spec.package_name == name and spec.kind is expected_kind:
            return spec
    raise CompilationError(f"Unknown {expected_kind.value} Skill: {name}")


def _validate_binding_names(
    slot: SlotSpec,
    names: Sequence[str],
) -> tuple[str, ...]:
    """校验单个槽位的组件名称类型、唯一性和数量范围。"""
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise CompilationError(f"Binding for slot '{slot.name}' must be a sequence")
    normalized = tuple(names)
    if not all(isinstance(name, str) for name in normalized):
        raise CompilationError(f"Binding for slot '{slot.name}' must contain names")
    if len(normalized) != len(set(normalized)):
        raise CompilationError(f"Binding for slot '{slot.name}' contains duplicates")
    if not slot.min_count <= len(normalized) <= slot.max_count:
        raise CompilationError(
            f"Binding count for slot '{slot.name}' must be between "
            f"{slot.min_count} and {slot.max_count}"
        )
    return normalized


def _component_matches_slot(component: RAGSkillSpec, slot: SlotSpec) -> bool:
    """判断组件 capability 和输入输出类型是否满足 Agentic 槽位。"""
    return any(
        capability.name == slot.capability
        and capability.input_type == slot.input_type
        and capability.output_type == slot.output_type
        for capability in component.provides
    )
