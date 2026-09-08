"""逐样本执行 RAG、评估，并让 Policy Model 优化一个 Skill 工作副本。"""

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

from framework import (
    EmbeddingClient,
    ExampleMetrics,
    GenerationEvaluator,
    ModelAPIError,
    ModelClient,
    RuntimeComponentContext,
    compile_rag_command,
    create_clients_from_config,
    embedding_service_fingerprint,
    evaluate_rag_result,
    select_rag_plan,
    summarize_metrics,
)

from .audit import compact_optimization_result, write_optimization_audit
from .config import (
    SelfOptimizationConfig,
    create_optimizer_model,
    load_self_optimization_config,
)
from .console import configure_console_utf8
from .optimizer import OptimizationError, OptimizationNoChange, SkillOptimizer
from .workspace import RevisionRejected, SkillWorkspace


class SelfOptimizationError(RuntimeError):
    """表示自优化数据、执行结果或运行参数不满足实验要求。"""


@dataclass(slots=True)
class OptimizationEventLogger:
    """把自优化阶段按顺序追加为 JSON Lines 事件。"""

    path: Path
    run_id: str
    session_id: str = field(default_factory=lambda: uuid4().hex)

    def write(self, event: str, **payload: Any) -> None:
        """写入一条带时间、运行 ID 和会话 ID 的结构化事件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_self_optimization(
    config: SelfOptimizationConfig,
    *,
    executor_model: ModelClient | None = None,
    embedding_model: EmbeddingClient | None = None,
    optimizer_model: ModelClient | None = None,
    generation_evaluator: GenerationEvaluator | None = None,
    max_examples: int | None = None,
    run_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """运行逐样本检索、XRAG 对齐评估和单 Skill 副本优化循环。"""
    framework_config = config.framework
    demo = framework_config.demo
    if demo is None:
        raise SelfOptimizationError(
            "Referenced framework config must contain a demo section"
        )
    effective_limit = (
        max_examples if max_examples is not None else config.optimization.max_examples
    )
    if effective_limit is not None and (
        isinstance(effective_limit, bool)
        or not isinstance(effective_limit, int)
        or effective_limit <= 0
    ):
        raise SelfOptimizationError("max_examples must be positive or null")

    workspace = SkillWorkspace.create(
        framework_config.skill_root,
        config.runs_root,
        run_id=run_id or config.run_id,
    )
    event_log = OptimizationEventLogger(
        workspace.run_root / "events.jsonl",
        run_id=workspace.run_id,
    )
    event_log.write(
        "run_started",
        config_path=str(config.config_path),
        framework_config_path=str(framework_config.config_path),
        source_skill_root=str(workspace.source_root),
        working_skill_root=str(workspace.skill_root),
        optimizer_model=config.optimizer.model,
        max_examples=effective_limit,
    )

    if executor_model is None:
        executor_model, configured_embedding = create_clients_from_config(
            framework_config
        )
        if embedding_model is None:
            embedding_model = configured_embedding
    if optimizer_model is None:
        optimizer_model = create_optimizer_model(config)
    optimizer = SkillOptimizer(optimizer_model, config.optimization)
    runtime_context = RuntimeComponentContext(
        executor_model=executor_model,
        embedding_model=embedding_model,
        vector_index_cache_dir=(
            framework_config.vector_index.cache_dir
            if framework_config.vector_index is not None
            else None
        ),
        embedding_fingerprint=(
            embedding_service_fingerprint(framework_config.embedding)
            if framework_config.embedding is not None and embedding_model is not None
            else None
        ),
    )

    corpus_records = _load_jsonl(demo.corpus_path)
    corpus = _index_corpus(corpus_records)
    test_records = _load_jsonl(demo.test_path)
    selected_examples = (
        test_records if effective_limit is None else test_records[:effective_limit]
    )
    if not selected_examples:
        raise SelfOptimizationError("Self-optimization test set is empty")
    event_log.write(
        "dataset_loaded",
        corpus_documents=len(corpus_records),
        test_examples=len(test_records),
        selected_examples=len(selected_examples),
        candidate_documents_only=demo.candidate_documents_only,
    )

    metrics_history: list[ExampleMetrics] = []
    output_examples: list[dict[str, Any]] = []
    report_path = workspace.run_root / "report.json"
    total = len(selected_examples)
    for index, example in enumerate(selected_examples, start=1):
        question_id = _required_text(example, "id")
        question = _required_text(example, "question")
        gold_answers = _answer_list(example)
        relevant_ids = _string_list(example, "relevant_document_ids")
        documents = _select_documents(
            example,
            corpus_records=corpus_records,
            corpus=corpus,
            candidate_documents_only=demo.candidate_documents_only,
        )
        request = {
            **framework_config.request_defaults,
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
        )

        plan = select_rag_plan(
            request,
            model=executor_model,
            skill_root=workspace.skill_root,
            manage_skill=framework_config.manage_skill,
        )
        command = compile_rag_command(
            plan,
            skill_root=workspace.skill_root,
            context=runtime_context,
        )
        result = command.run(request)
        result["selection"] = plan.to_dict()
        result["compiled_instruction"] = command.instruction
        event_log.write(
            "execution_completed",
            question_id=question_id,
            selection=plan.to_dict(),
            trace=result.get("trace", []),
            component_timings=result.get("component_timings", []),
            retrieved_document_ids=_retrieved_ids(result),
            prediction=result.get("answer"),
        )

        metrics = evaluate_rag_result(
            result,
            gold_answers=gold_answers,
            relevant_ids=relevant_ids,
            generation_evaluator=generation_evaluator,
        )
        metrics_history.append(metrics)
        running_summary = summarize_metrics(metrics_history).to_dict()
        evaluation = {
            "question_id": question_id,
            "gold_answers": gold_answers,
            "metrics": metrics.to_dict(),
            "running_summary": running_summary,
        }
        event_log.write("evaluation_completed", **evaluation)

        candidate_names = _participating_skills(plan.to_dict())
        retrieval_process = _build_retrieval_process(
            question=question,
            result=result,
            max_documents=config.optimization.max_documents,
            excerpt_chars=config.optimization.document_excerpt_chars,
        )
        event_log.write(
            "optimization_requested",
            question_id=question_id,
            candidate_skills=list(candidate_names),
        )
        optimization_record: dict[str, Any]
        proposal = None
        optimization_error: Exception | None = None
        try:
            proposal = optimizer.propose(
                workspace=workspace,
                candidate_names=candidate_names,
                evaluation=evaluation,
                retrieval_process=retrieval_process,
            )
            revision = workspace.apply_proposal(
                proposal,
                candidate_names=candidate_names,
                max_edits=config.optimization.max_edits,
                max_file_chars=config.optimization.max_skill_file_chars,
                allow_new_files=config.optimization.allow_new_files,
            )
            optimization_record = {
                "status": "accepted",
                "selected_skill": proposal.selected_skill,
                "rationale": proposal.rationale,
                "revision": revision.to_dict(),
            }
        except OptimizationNoChange as exc:
            optimization_record = {
                "status": "skipped",
                "selected_skill": exc.selected_skill,
                "rationale": exc.rationale,
                "reason": _safe_error_message(exc),
            }
        except (ModelAPIError, OptimizationError, RevisionRejected) as exc:
            optimization_error = exc
            optimization_record = {
                "status": "rejected",
                "error_type": type(exc).__name__,
                "error": _safe_error_message(exc),
            }
        audit = write_optimization_audit(
            workspace.run_root / "optimizer_calls" / f"example-{index:04d}",
            model_calls=optimizer.calls,
            proposal=proposal,
            result=optimization_record,
        )
        optimization_record["audit"] = audit
        event_log.write(
            f"optimization_{optimization_record['status']}",
            question_id=question_id,
            **optimization_record,
        )
        event_log.write(
            "optimization_audit_saved",
            question_id=question_id,
            **audit,
        )
        if (
            optimization_error is not None
            and not config.optimization.continue_on_optimization_error
        ):
            raise optimization_error

        output_examples.append(
            {
                "id": question_id,
                "question": question,
                "gold_answers": gold_answers,
                "prediction": result.get("answer"),
                "retrieved_document_ids": _retrieved_ids(result),
                "selection": plan.to_dict(),
                "trace": result.get("trace", []),
                "component_timings": result.get("component_timings", []),
                "metrics": metrics.to_dict(),
                "running_summary": running_summary,
                "optimization": optimization_record,
            }
        )
        _write_json(
            report_path,
            _build_report(
                config=config,
                workspace=workspace,
                examples=output_examples,
                summary=running_summary,
                event_log=event_log,
            ),
        )
        if verbose:
            print(f"\n[{index}/{total}] {question_id}")
            print("Question:", question)
            print("Prediction:", result.get("answer"))
            print("Metrics:", json.dumps(metrics.to_dict(), ensure_ascii=False))
            print("Running Summary:", json.dumps(running_summary, ensure_ascii=False))
            print(
                "Optimization:",
                json.dumps(
                    compact_optimization_result(optimization_record),
                    ensure_ascii=False,
                ),
            )

    workspace.assert_source_unchanged()
    summary = summarize_metrics(metrics_history).to_dict()
    report = _build_report(
        config=config,
        workspace=workspace,
        examples=output_examples,
        summary=summary,
        event_log=event_log,
    )
    _write_json(report_path, report)
    event_log.write(
        "run_completed",
        report_path=str(report_path),
        working_skill_root=str(workspace.skill_root),
        summary=summary,
    )
    if verbose:
        print("Summary:", json.dumps(summary, ensure_ascii=False))
        print("Working Skill Copy:", workspace.skill_root)
        print("Report:", report_path)
        print("Log:", event_log.path)
    return report


def parse_args() -> argparse.Namespace:
    """解析自优化配置、样本上限和可选运行标识。"""
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation and optimize an isolated Skill copy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("settings.yaml"),
        help="Self-optimization YAML config path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Temporarily override optimization.max_examples.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional unique run directory name.",
    )
    return parser.parse_args()


def main() -> None:
    """加载本地配置并启动一次 Skill 副本自优化实验。"""
    configure_console_utf8()
    args = parse_args()
    config = load_self_optimization_config(args.config)
    run_self_optimization(config, max_examples=args.limit, run_id=args.run_id)


def _build_retrieval_process(
    *,
    question: str,
    result: Mapping[str, Any],
    max_documents: int,
    excerpt_chars: int,
) -> dict[str, Any]:
    """提取选择、检索轨迹、耗时和有界文档片段供优化模型分析。"""
    documents_value = result.get("documents", [])
    documents = []
    if isinstance(documents_value, Sequence) and not isinstance(
        documents_value,
        (str, bytes, bytearray),
    ):
        for document in documents_value[:max_documents]:
            if not isinstance(document, Mapping):
                continue
            text = str(document.get("text", ""))
            documents.append(
                {
                    "id": document.get("id"),
                    "title": document.get("title"),
                    "score": document.get("score"),
                    "text_excerpt": text[:excerpt_chars],
                    "text_truncated": len(text) > excerpt_chars,
                }
            )
    return {
        "question": question,
        "prediction": result.get("answer"),
        "selection": result.get("selection", {}),
        "compiled_instruction": result.get("compiled_instruction"),
        "trace": result.get("trace", []),
        "component_timings": result.get("component_timings", []),
        "retrieved_documents": documents,
    }


def _participating_skills(selection: Mapping[str, Any]) -> tuple[str, ...]:
    """从本轮选择计划中按 Manage、Agentic、Component 顺序收集候选。"""
    names = [
        _required_text(selection, "manage_skill"),
        _required_text(selection, "agentic_skill"),
    ]
    bindings = selection.get("component_bindings")
    if not isinstance(bindings, Mapping):
        raise SelfOptimizationError("selection.component_bindings must be a mapping")
    for values in bindings.values():
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values,
            Sequence,
        ):
            raise SelfOptimizationError("Component bindings must be string lists")
        names.extend(str(value) for value in values)
    return tuple(dict.fromkeys(names))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取实验 JSONL 并验证每个非空行都是对象。"""
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SelfOptimizationError(f"Cannot read data {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SelfOptimizationError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise SelfOptimizationError(
                f"Data record at {path}:{line_number} must be an object"
            )
        records.append(dict(value))
    return records


def _index_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """按唯一文档 ID 建立语料索引并验证正文存在。"""
    corpus = {}
    for record in records:
        document_id = _required_text(record, "id")
        if document_id in corpus:
            raise SelfOptimizationError(f"Duplicate corpus document ID: {document_id}")
        if not isinstance(record.get("text"), str):
            raise SelfOptimizationError(f"Corpus document '{document_id}' has no text")
        corpus[document_id] = dict(record)
    if not corpus:
        raise SelfOptimizationError("Corpus is empty")
    return corpus


def _select_documents(
    example: Mapping[str, Any],
    *,
    corpus_records: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]],
    candidate_documents_only: bool,
) -> list[dict[str, Any]]:
    """依据基础 Demo 设置选择单题候选文档或完整共享语料。"""
    if not candidate_documents_only:
        return [dict(document) for document in corpus_records]
    candidate_ids = _string_list(example, "candidate_document_ids")
    missing = [document_id for document_id in candidate_ids if document_id not in corpus]
    if missing:
        raise SelfOptimizationError(f"Candidate documents are missing: {missing}")
    return [dict(corpus[document_id]) for document_id in candidate_ids]


def _answer_list(example: Mapping[str, Any]) -> list[str]:
    """读取一个或多个标准答案，并兼容单 answer 字段。"""
    if "answers" in example:
        return _string_list(example, "answers")
    return [_required_text(example, "answer")]


def _retrieved_ids(result: Mapping[str, Any]) -> list[str]:
    """从执行结果中提取有序检索文档 ID。"""
    documents = result.get("documents")
    if isinstance(documents, (str, bytes, bytearray)) or not isinstance(
        documents,
        Sequence,
    ):
        raise SelfOptimizationError("RAG result.documents must be a sequence")
    identifiers = []
    for index, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise SelfOptimizationError(
                f"RAG result.documents[{index}] must be an object"
            )
        identifiers.append(_required_text(document, "id"))
    return identifiers


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """读取对象中的必需非空字符串。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SelfOptimizationError(f"'{key}' must be a non-empty string")
    return value.strip()


def _string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    """读取至少包含一个非空字符串的序列字段。"""
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SelfOptimizationError(f"'{key}' must be a sequence of strings")
    if not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise SelfOptimizationError(f"'{key}' must contain non-empty strings")
    return [item.strip() for item in value]


def _build_report(
    *,
    config: SelfOptimizationConfig,
    workspace: SkillWorkspace,
    examples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    event_log: OptimizationEventLogger,
) -> dict[str, Any]:
    """构造不含密钥和完整语料的可提交实验报告。"""
    return {
        "schema_version": 1,
        "run_id": workspace.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source_skill_root": str(workspace.source_root),
        "source_skill_hash": workspace.source_hash,
        "working_skill_root": str(workspace.skill_root),
        "optimizer": {
            "provider": config.optimizer.provider,
            "model": config.optimizer.model,
            "base_url": config.optimizer.base_url,
        },
        "summary": dict(summary),
        "examples": [dict(example) for example in examples],
        "artifacts": {
            "report_path": str(workspace.run_root / "report.json"),
            "log_path": str(event_log.path),
            "revisions_root": str(workspace.revisions_root),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """创建父目录并写入格式化 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_error_message(error: Exception) -> str:
    """在日志前移除可能出现的 OpenAI 风格密钥。"""
    return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", str(error))


if __name__ == "__main__":
    main()
