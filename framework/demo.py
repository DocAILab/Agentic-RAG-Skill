"""从统一配置运行 HotpotQA demo 并保存逐题结果与汇总指标。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .compiler import CompiledRAGCommand, RuntimeComponentContext, compile_rag_command
from .config import (
    FrameworkConfig,
    create_clients_from_config,
    embedding_service_fingerprint,
    load_framework_config,
)
from .evaluation import EvaluationExample, evaluate_batch, evaluate_example
from .models import EmbeddingClient, ModelClient
from .selection import (
    RAGSelectionPlan,
    run_manage_stage,
    select_agentic_skill,
    select_component_skills,
)


class DemoError(ValueError):
    """表示 demo 配置、数据或运行结果不满足入口契约。"""


@dataclass(slots=True)
class DemoEventLogger:
    """把 demo 中间阶段作为可追加的 JSON Lines 事件写入日志。"""

    path: Path
    run_id: str = field(default_factory=lambda: uuid4().hex)

    def write(self, event: str, **payload: Any) -> None:
        """追加一条带时间与运行标识的结构化日志事件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _select_and_compile(
    request: Mapping[str, Any],
    *,
    config: FrameworkConfig,
    model: ModelClient,
    runtime_context: RuntimeComponentContext,
    event_log: DemoEventLogger,
    selection_scope: str,
    question_id: str | None = None,
) -> tuple[RAGSelectionPlan, CompiledRAGCommand]:
    """执行一次三级 Skill 选择、记录阶段日志并编译可复用命令。"""
    identity = {
        "selection_scope": selection_scope,
        **({"question_id": question_id} if question_id is not None else {}),
    }
    manage_result = run_manage_stage(
        request,
        model=model,
        skill_root=config.skill_root,
        manage_skill=config.manage_skill,
    )
    event_log.write(
        "manage_completed",
        **identity,
        manage_skill=manage_result.manage_skill,
        guidance=manage_result.guidance,
        reason=manage_result.reason,
    )
    agentic_result = select_agentic_skill(
        request,
        manage_result=manage_result,
        model=model,
        skill_root=config.skill_root,
    )
    event_log.write(
        "agentic_selected",
        **identity,
        agentic_skill=agentic_result.spec.package_name,
        reason=agentic_result.reason,
    )
    component_result = select_component_skills(
        request,
        agentic_result=agentic_result,
        model=model,
        skill_root=config.skill_root,
    )
    event_log.write(
        "components_selected",
        **identity,
        component_bindings={
            slot: list(names) for slot, names in component_result.bindings.items()
        },
        reason=component_result.reason,
    )
    plan = RAGSelectionPlan(
        manage_skill=manage_result.manage_skill,
        manage_guidance=manage_result.guidance,
        manage_reason=manage_result.reason,
        agentic_skill=agentic_result.spec.package_name,
        agentic_reason=agentic_result.reason,
        component_bindings=component_result.bindings,
        component_reason=component_result.reason,
    )
    command = compile_rag_command(
        plan,
        skill_root=config.skill_root,
        context=runtime_context,
    )
    event_log.write(
        "command_compiled",
        **identity,
        instruction=command.instruction,
    )
    return plan, command


def _sample_batch_queries(
    examples: Sequence[Mapping[str, Any]],
    sample_size: int,
) -> list[str]:
    """沿批次顺序均匀抽取问题文本，确定性覆盖整个测试范围。"""
    total = len(examples)
    if total <= sample_size:
        sampled_examples = examples
    elif sample_size == 1:
        sampled_examples = [examples[total // 2]]
    else:
        indices = [
            round(position * (total - 1) / (sample_size - 1))
            for position in range(sample_size)
        ]
        sampled_examples = [examples[index] for index in indices]
    return [_required_text(example, "question") for example in sampled_examples]


def run_demo(
    config: FrameworkConfig,
    *,
    model: ModelClient | None = None,
    embedding_model: EmbeddingClient | None = None,
    max_examples: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """加载配置中的数据并执行检索、生成、测评和结果落盘。"""
    demo = config.demo
    if demo is None:
        raise DemoError("Framework config does not contain a demo section")
    if max_examples is not None and (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples <= 0
    ):
        raise DemoError("max_examples override must be positive")

    event_log = DemoEventLogger(demo.log_path)
    event_log.write(
        "run_started",
        config_path=str(config.config_path),
        corpus_path=str(demo.corpus_path),
        test_path=str(demo.test_path),
        result_path=str(demo.result_path),
        max_examples=(max_examples if max_examples is not None else demo.max_examples),
        candidate_documents_only=demo.candidate_documents_only,
        select_skills_per_example=demo.select_skills_per_example,
        batch_selection_query_sample_size=demo.batch_selection_query_sample_size,
        vector_index_cache_dir=(
            str(config.vector_index.cache_dir)
            if config.vector_index is not None
            else None
        ),
        request=dict(demo.request),
    )

    try:
        if model is None:
            model, configured_embedding = create_clients_from_config(config)
            if embedding_model is None:
                embedding_model = configured_embedding
    except Exception as exc:
        event_log.write(
            "run_failed",
            error_type=type(exc).__name__,
            error=_safe_error_message(exc),
        )
        raise
    runtime_context = RuntimeComponentContext(
        executor_model=model,
        embedding_model=embedding_model,
        vector_index_cache_dir=(
            config.vector_index.cache_dir if config.vector_index is not None else None
        ),
        embedding_fingerprint=(
            embedding_service_fingerprint(config.embedding)
            if config.embedding is not None
            else None
        ),
    )

    corpus_records = _load_jsonl(demo.corpus_path)
    test_records = _load_jsonl(demo.test_path)
    corpus = _index_corpus(corpus_records)
    effective_limit = max_examples if max_examples is not None else demo.max_examples
    selected_tests = (
        test_records if effective_limit is None else test_records[:effective_limit]
    )
    if not selected_tests:
        raise DemoError("Demo test set contains no selected examples")
    event_log.write(
        "dataset_loaded",
        corpus_documents=len(corpus_records),
        test_examples=len(test_records),
        selected_examples=len(selected_tests),
    )

    evaluation_examples: list[EvaluationExample] = []
    output_examples: list[dict[str, Any]] = []
    total = len(selected_tests)
    batch_plan: RAGSelectionPlan | None = None
    batch_command: CompiledRAGCommand | None = None
    if not demo.select_skills_per_example:
        sampled_queries = _sample_batch_queries(
            selected_tests,
            demo.batch_selection_query_sample_size,
        )
        batch_request = {
            **config.request_defaults,
            **demo.request,
            "query": "Select one reusable RAG workflow for this evaluation batch.",
            "sampled_queries": sampled_queries,
            "query_count": total,
            "sampled_query_count": len(sampled_queries),
            "documents": corpus_records,
        }
        try:
            batch_plan, batch_command = _select_and_compile(
                batch_request,
                config=config,
                model=model,
                runtime_context=runtime_context,
                event_log=event_log,
                selection_scope="batch",
            )
        except Exception as exc:
            event_log.write(
                "run_failed",
                selection_scope="batch",
                error_type=type(exc).__name__,
                error=_safe_error_message(exc),
            )
            raise

    for index, example in enumerate(selected_tests, start=1):
        question_id = _required_text(example, "id")
        question = _required_text(example, "question")
        answers = _answer_list(example)
        relevant_ids = _string_list(example, "relevant_document_ids")
        documents = _select_documents(
            example,
            corpus_records=corpus_records,
            corpus=corpus,
            candidate_documents_only=demo.candidate_documents_only,
        )
        request = {
            **config.request_defaults,
            **demo.request,
            "query": question,
            "documents": documents,
        }
        event_log.write(
            "example_started",
            index=index,
            total=total,
            question_id=question_id,
            question=question,
            gold_answers=answers,
            relevant_document_ids=relevant_ids,
            input_document_ids=[document["id"] for document in documents],
        )
        try:
            if demo.select_skills_per_example:
                plan, command = _select_and_compile(
                    request,
                    config=config,
                    model=model,
                    runtime_context=runtime_context,
                    event_log=event_log,
                    selection_scope="example",
                    question_id=question_id,
                )
            else:
                assert batch_plan is not None and batch_command is not None
                plan, command = batch_plan, batch_command
            result = command.run(request)
            result["selection"] = plan.to_dict()
            result["compiled_instruction"] = command.instruction
        except Exception as exc:
            event_log.write(
                "example_failed",
                question_id=question_id,
                error_type=type(exc).__name__,
                error=_safe_error_message(exc),
            )
            event_log.write(
                "run_failed",
                question_id=question_id,
                error_type=type(exc).__name__,
                error=_safe_error_message(exc),
            )
            raise
        retrieved_ids = _retrieved_ids(result)
        evaluation = EvaluationExample(
            prediction=_required_text(result, "answer"),
            gold_answers=answers,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
        )
        metrics = evaluate_example(evaluation)
        evaluation_examples.append(evaluation)
        running_summary = evaluate_batch(evaluation_examples).to_dict()
        selection_value = result.get("selection", {})
        selection = (
            dict(selection_value) if isinstance(selection_value, Mapping) else {}
        )
        event_log.write(
            "execution_completed",
            question_id=question_id,
            retrieved_document_ids=retrieved_ids,
            prediction=result["answer"],
            trace=result.get("trace", []),
            component_timings=result.get("component_timings", []),
            embedding_cache=runtime_context.embedding_cache_info(),
            vector_index_cache=runtime_context.vector_index_cache_info(),
            compiled_instruction=result.get("compiled_instruction"),
        )
        event_log.write(
            "evaluation_completed",
            question_id=question_id,
            metrics=metrics.to_dict(),
            running_summary=running_summary,
        )
        output_examples.append(
            {
                "id": question_id,
                "question": question,
                "gold_answers": answers,
                "prediction": result["answer"],
                "metrics": metrics.to_dict(),
                "retrieved_document_ids": retrieved_ids,
                "relevant_document_ids": relevant_ids,
                "selection": selection,
                "trace": result.get("trace", []),
                "component_timings": result.get("component_timings", []),
                "vector_index_cache": runtime_context.vector_index_cache_info(),
                "compiled_instruction": result.get("compiled_instruction"),
            }
        )
        if verbose:
            print(f"\n[{index}/{total}] {question_id}")
            print("Question:", question)
            print("Prediction:", result["answer"])
            print("Agentic Skill:", selection.get("agentic_skill"))
            print(
                "Component Bindings:",
                json.dumps(
                    selection.get("component_bindings", {}),
                    ensure_ascii=False,
                ),
            )
            print("Metrics:", json.dumps(metrics.to_dict(), ensure_ascii=False))
            print(
                "Component Timings:",
                json.dumps(result.get("component_timings", []), ensure_ascii=False),
            )
            print(
                "Vector Index Cache:",
                json.dumps(
                    runtime_context.vector_index_cache_info(),
                    ensure_ascii=False,
                ),
            )
            print("Running Summary:", json.dumps(running_summary, ensure_ascii=False))

    summary = evaluate_batch(evaluation_examples)
    report = {
        "schema_version": 1,
        "run_id": event_log.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "corpus_path": str(demo.corpus_path),
            "test_path": str(demo.test_path),
            "candidate_documents_only": demo.candidate_documents_only,
            "select_skills_per_example": demo.select_skills_per_example,
            "batch_selection_query_sample_size": (
                demo.batch_selection_query_sample_size
            ),
            "vector_index_cache_dir": (
                str(config.vector_index.cache_dir)
                if config.vector_index is not None
                else None
            ),
        },
        "artifacts": {
            "result_path": str(demo.result_path),
            "log_path": str(demo.log_path),
        },
        "summary": summary.to_dict(),
        "batch_selection": (
            batch_plan.to_dict() if batch_plan is not None else None
        ),
        "examples": output_examples,
    }
    _write_report(demo.result_path, report)
    event_log.write(
        "run_completed",
        result_path=str(demo.result_path),
        log_path=str(demo.log_path),
        summary=summary.to_dict(),
    )
    if verbose:
        print("Summary:", json.dumps(summary.to_dict(), ensure_ascii=False))
        print("Results:", demo.result_path)
        print("Log:", demo.log_path)
    return report


def parse_args() -> argparse.Namespace:
    """解析可选配置路径和临时样本数量覆盖值。"""
    parser = argparse.ArgumentParser(
        description="Run the configured hierarchical RAG demo."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("settings.yaml"),
        help="Framework YAML config path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Temporarily override demo.max_examples.",
    )
    return parser.parse_args()


def main() -> None:
    """加载 YAML 配置并执行一次命令行 demo。"""
    args = parse_args()
    config = load_framework_config(args.config)
    run_demo(config, max_examples=args.limit)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 并校验每一行都是 JSON 对象。"""
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DemoError(f"Cannot read demo data {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DemoError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, Mapping):
            raise DemoError(f"Demo record at {path}:{line_number} must be an object")
        records.append(dict(record))
    return records


def _index_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """按唯一文档 ID 建立 corpus 索引。"""
    corpus: dict[str, dict[str, Any]] = {}
    for record in records:
        document_id = _required_text(record, "id")
        if document_id in corpus:
            raise DemoError(f"Duplicate corpus document ID: {document_id}")
        if not isinstance(record.get("text"), str):
            raise DemoError(f"Corpus document '{document_id}' has no text")
        corpus[document_id] = dict(record)
    if not corpus:
        raise DemoError("Demo corpus is empty")
    return corpus


def _select_documents(
    example: Mapping[str, Any],
    *,
    corpus_records: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]],
    candidate_documents_only: bool,
) -> list[dict[str, Any]]:
    """按配置选择每题候选文档或完整共享小语料。"""
    if not candidate_documents_only:
        return [dict(document) for document in corpus_records]
    candidate_ids = _string_list(example, "candidate_document_ids")
    missing = [document_id for document_id in candidate_ids if document_id not in corpus]
    if missing:
        raise DemoError(f"Candidate documents are missing from corpus: {missing}")
    return [dict(corpus[document_id]) for document_id in candidate_ids]


def _answer_list(example: Mapping[str, Any]) -> list[str]:
    """读取标准答案别名，并兼容只有单个 answer 的样本。"""
    if "answers" in example:
        return _string_list(example, "answers")
    return [_required_text(example, "answer")]


def _retrieved_ids(result: Mapping[str, Any]) -> list[str]:
    """从 framework 结果中读取有序检索文档 ID。"""
    documents = result.get("documents")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents, Sequence
    ):
        raise DemoError("RAG result.documents must be a sequence")
    identifiers = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise DemoError(f"RAG result.documents[{index}] must be an object")
        identifiers.append(_required_text(document, "id"))
    return identifiers


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取非空字符串字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DemoError(f"'{key}' must be a non-empty string")
    return value.strip()


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    """读取非空字符串列表字段。"""
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DemoError(f"'{key}' must be a sequence of strings")
    if not all(isinstance(item, str) for item in value):
        raise DemoError(f"'{key}' must contain only strings")
    normalized = [item.strip() for item in value]
    if not normalized or any(not item for item in normalized):
        raise DemoError(f"'{key}' must contain non-empty strings")
    return normalized


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    """创建结果目录并写入格式化 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_error_message(error: Exception) -> str:
    """移除错误文本中可能出现的 OpenAI 风格密钥。"""
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", str(error))


if __name__ == "__main__":
    main()
