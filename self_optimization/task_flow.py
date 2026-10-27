"""在同一个 Skill 工作副本上顺序执行多个 RAG task 并逐 task 优化。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from framework.data_groups import DataGroup, load_group

from framework import (
    EmbeddingClient,
    ExampleMetrics,
    GenerationEvaluator,
    ModelAPIError,
    ModelClient,
    RuntimeComponentContext,
    build_rag_request,
    compile_rag_command,
    create_clients_from_config,
    embedding_service_fingerprint,
    evaluate_rag_result,
    select_rag_plan,
    summarize_metrics,
)

from .audit import compact_optimization_result, write_optimization_audit
from .config import create_optimizer_model
from .console import configure_console_utf8
from .optimizer import OptimizationError, OptimizationNoChange, SkillOptimizer
from .runner import (
    OptimizationEventLogger,
    _build_retrieval_process,
    _participating_skills,
    _retrieved_ids,
    _safe_error_message,
    _write_json,
)
from .task_config import (
    RAGTaskConfig,
    TaskWorkflowConfig,
    load_task_workflow_config,
)
from .workspace import RevisionRejected, SkillWorkspace


class TaskWorkflowError(RuntimeError):
    """表示 task 数据或顺序执行结果不符合任务流契约。"""


@dataclass(frozen=True, slots=True)
class TaskExample:
    """保存一个可直接执行和评估的本地 RAG task 样本。"""

    id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    documents: tuple[Mapping[str, Any], ...]
    relevant_document_ids: tuple[str, ...]
    label_type: str | None = None
    group_id: str | None = None
    task_id: str | None = None


def run_task_workflow(
    config: TaskWorkflowConfig,
    *,
    executor_model: ModelClient | None = None,
    embedding_model: EmbeddingClient | None = None,
    optimizer_model: ModelClient | None = None,
    generation_evaluator: GenerationEvaluator | None = None,
    max_tasks: int | None = None,
    examples_per_task: int | None = None,
    run_id: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """依次运行所有 task，并在每个 task 汇总评估后优化一次 Skill。"""
    if not isinstance(config, TaskWorkflowConfig):
        raise TaskWorkflowError(
            "Expected TaskWorkflowConfig; use run_sample_optimization for the legacy sample runner"
        )
    selected = _select_task_examples(
        config.tasks, group=config.group, max_tasks=max_tasks,
        examples_per_task=examples_per_task,
    )
    selection = _selection_summary(selected, max_tasks, examples_per_task)
    base = config.self_optimization
    framework_config = base.framework
    workspace = SkillWorkspace.create(
        framework_config.skill_root,
        base.runs_root,
        run_id=run_id or base.run_id,
    )
    event_log = OptimizationEventLogger(
        workspace.run_root / "task-flow.events.jsonl",
        run_id=workspace.run_id,
    )
    event_log.write(
        "workflow_started",
        task_flow_config=str(config.config_path),
        source_skill_root=str(workspace.source_root),
        working_skill_root=str(workspace.skill_root),
        group_id=config.group.group_id if config.group is not None else None,
        task_names=[task.name for task, _ in selected],
        selection=selection,
        optimizer_model=base.optimizer.model,
    )

    if executor_model is None:
        executor_model, configured_embedding = create_clients_from_config(
            framework_config
        )
        if embedding_model is None:
            embedding_model = configured_embedding
    if optimizer_model is None:
        optimizer_model = create_optimizer_model(base)
    optimizer = SkillOptimizer(optimizer_model, base.optimization)
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
    if config.group is not None:
        shared_documents = runtime_context.bind_corpus(
            config.group.documents, config.group.corpus_fingerprint
        )
        if shared_documents is not config.group.documents:
            selected = tuple(
                (task, tuple(replace(example, documents=shared_documents) for example in examples))
                for task, examples in selected
            )

    report_path = workspace.run_root / "task-flow.report.json"
    task_records = []
    overall_metrics: list[ExampleMetrics] = []
    for task_index, (task, task_examples) in enumerate(selected, start=1):
        task_identity = {
            "task_id": task.name,
            "group_id": config.group.group_id if config.group is not None else None,
        }
        event_log.write(
            "task_started",
            **task_identity,
            task_index=task_index,
            total_tasks=len(selected),
            task_name=task.name,
            dataset=task.dataset,
            examples=len(task_examples),
            skill_hash_before=workspace.working_hash(),
        )
        if verbose:
            print(f"\n=== Task {task_index}/{len(selected)}: {task.name} ===")

        executions = []
        candidate_names = []
        for example_index, example in enumerate(task_examples, start=1):
            request = _execution_request(framework_config, task, example)
            event_log.write(
                "task_example_started",
                **task_identity,
                task_name=task.name,
                example_index=example_index,
                total_examples=len(task_examples),
                question_id=example.id,
                label_type=example.label_type,
                question=example.question,
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
            candidate_names.extend(_participating_skills(plan.to_dict()))
            executions.append((example, result))
            event_log.write(
                "task_example_executed",
                **task_identity,
                task_name=task.name,
                question_id=example.id,
                label_type=example.label_type,
                selection=plan.to_dict(),
                prediction=result.get("answer"),
                retrieved_document_ids=_retrieved_ids(result),
                trace=result.get("trace", []),
                component_timings=result.get("component_timings", []),
            )
            if verbose:
                print(
                    f"[{example_index}/{len(task_examples)}] {example.id}: "
                    f"{result.get('answer')}"
                )

        task_metrics = []
        example_records = []
        process_examples = []
        for example, result in executions:
            metrics = evaluate_rag_result(
                result,
                gold_answers=example.answers,
                relevant_ids=example.relevant_document_ids,
                generation_evaluator=generation_evaluator,
            )
            task_metrics.append(metrics)
            overall_metrics.append(metrics)
            example_records.append(
                {
                    "id": example.id,
                    "group_id": example.group_id,
                    "task_id": example.task_id or task.name,
                    "label_type": example.label_type,
                    "question": example.question,
                    "gold_answers": list(example.answers),
                    "prediction": result.get("answer"),
                    "retrieved_document_ids": _retrieved_ids(result),
                    "selection": result.get("selection", {}),
                    "metrics": metrics.to_dict(),
                    "trace": result.get("trace", []),
                    "component_timings": result.get("component_timings", []),
                }
            )
            if len(process_examples) < config.max_process_examples:
                process_examples.append(
                    _build_retrieval_process(
                        question=example.question,
                        result=result,
                        max_documents=base.optimization.max_documents,
                        excerpt_chars=base.optimization.document_excerpt_chars,
                    )
                )

        task_summary = summarize_metrics(task_metrics).to_dict()
        overall_summary = summarize_metrics(overall_metrics).to_dict()
        task_evaluation = {
            "task_name": task.name,
            "task_id": task.name,
            "group_id": config.group.group_id if config.group is not None else None,
            "dataset": task.dataset,
            "example_count": len(example_records),
            "task_summary": task_summary,
            "overall_summary": overall_summary,
            "examples": [
                {
                    "id": record["id"],
                    "label_type": record["label_type"],
                    "metrics": record["metrics"],
                }
                for record in example_records
            ],
        }
        event_log.write("task_evaluation_completed", **task_evaluation)
        unique_candidates = tuple(dict.fromkeys(candidate_names))
        event_log.write(
            "task_optimization_requested",
            **task_identity,
            task_name=task.name,
            candidate_skills=list(unique_candidates),
        )
        proposal = None
        optimization_error: Exception | None = None
        try:
            proposal = optimizer.propose(
                workspace=workspace,
                candidate_names=unique_candidates,
                evaluation=task_evaluation,
                retrieval_process={
                    "task_name": task.name,
                    "dataset": task.dataset,
                    "executed_examples": len(executions),
                    "included_process_examples": len(process_examples),
                    "examples": process_examples,
                },
            )
            revision = workspace.apply_proposal(
                proposal,
                candidate_names=unique_candidates,
                max_edits=base.optimization.max_edits,
                max_file_chars=base.optimization.max_skill_file_chars,
                allow_new_files=base.optimization.allow_new_files,
            )
            verification = None
            if base.optimization.verify_revision_execution:
                try:
                    verification = _verify_task_revision(
                        examples=task_examples,
                        baseline_metrics=task_metrics,
                        sample_count=config.validation_examples_per_task,
                        task=task,
                        selected_skill=proposal.selected_skill,
                        framework_config=framework_config,
                        workspace=workspace,
                        executor_model=executor_model,
                        runtime_context=runtime_context,
                        generation_evaluator=generation_evaluator,
                    )
                except Exception as exc:
                    reason = (
                        "Post-revision execution validation failed: "
                        f"{_safe_error_message(exc)}"
                    )
                    workspace.rollback_revision(revision, reason=reason)
                    raise RevisionRejected(reason) from exc
            optimization_record = {
                "status": "accepted",
                "selected_skill": proposal.selected_skill,
                "rationale": proposal.rationale,
                "revision": revision.to_dict(),
            }
            if verification is not None:
                optimization_record["verification"] = verification
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
            workspace.run_root / "optimizer_calls" / f"task-{task_index:04d}",
            model_calls=optimizer.calls,
            proposal=proposal,
            result=optimization_record,
        )
        optimization_record["audit"] = audit
        event_log.write(
            f"task_optimization_{optimization_record['status']}",
            **task_identity,
            task_name=task.name,
            **optimization_record,
        )
        event_log.write(
            "task_optimization_audit_saved",
            **task_identity,
            task_name=task.name,
            **audit,
        )
        if (
            optimization_error is not None
            and not base.optimization.continue_on_optimization_error
        ):
            raise optimization_error

        task_record = {
            "name": task.name,
            "task_id": task.name,
            "group_id": config.group.group_id if config.group is not None else None,
            "dataset": task.dataset,
            "data_path": str(task.data_path),
            "summary": task_summary,
            "overall_summary_after_task": overall_summary,
            "candidate_skills": list(unique_candidates),
            "examples": example_records,
            "optimization": optimization_record,
        }
        task_records.append(task_record)
        event_log.write(
            "task_completed",
            **task_identity,
            task_name=task.name,
            summary=task_summary,
            optimization_status=optimization_record["status"],
            skill_hash_after=workspace.working_hash(),
        )
        _write_json(
            report_path,
            _build_workflow_report(
                config=config,
                workspace=workspace,
                tasks=task_records,
                overall_summary=overall_summary,
                event_log=event_log,
                selection=selection,
            ),
        )
        if verbose:
            print("Task Summary:", json.dumps(task_summary, ensure_ascii=False))
            print(
                "Task Optimization:",
                json.dumps(
                    compact_optimization_result(optimization_record),
                    ensure_ascii=False,
                ),
            )

    workspace.assert_source_unchanged()
    final_summary = summarize_metrics(overall_metrics).to_dict()
    report = _build_workflow_report(
        config=config,
        workspace=workspace,
        tasks=task_records,
        overall_summary=final_summary,
        event_log=event_log,
        selection=selection,
    )
    _write_json(report_path, report)
    event_log.write(
        "workflow_completed",
        completed_tasks=len(task_records),
        report_path=str(report_path),
        working_skill_root=str(workspace.skill_root),
        overall_summary=final_summary,
    )
    if verbose:
        print("\nWorkflow Summary:", json.dumps(final_summary, ensure_ascii=False))
        print("Working Skill Copy:", workspace.skill_root)
        print("Report:", report_path)
        print("Log:", event_log.path)
    return report


def _verify_revision_execution(
    *,
    example: TaskExample,
    task: RAGTaskConfig,
    selected_skill: str,
    baseline_metrics: Mapping[str, Any],
    framework_config: Any,
    workspace: SkillWorkspace,
    executor_model: ModelClient,
    runtime_context: RuntimeComponentContext,
    generation_evaluator: GenerationEvaluator | None,
    enforce_improvement: bool = True,
) -> dict[str, Any]:
    """在接受 revision 前重跑真实样本，并要求其负责阶段的指标确有提升。"""
    request = _execution_request(framework_config, task, example)
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
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise TaskWorkflowError("Post-revision execution returned an empty answer")
    metrics = evaluate_rag_result(
        result,
        gold_answers=example.answers,
        relevant_ids=example.relevant_document_ids,
        generation_evaluator=generation_evaluator,
    ).to_dict()
    comparison = _compare_revision_metrics(
        selected_skill=selected_skill,
        baseline=baseline_metrics,
        revised=metrics,
        prediction=answer,
        gold_answers=example.answers,
    )
    if enforce_improvement and not comparison["improved"]:
        raise TaskWorkflowError(str(comparison["reason"]))
    return {
        "question_id": example.id,
        "group_id": example.group_id,
        "task_id": example.task_id or task.name,
        "label_type": example.label_type,
        "prediction": answer.strip(),
        "agentic_skill": plan.agentic_skill,
        "retrieved_document_ids": _retrieved_ids(result),
        "metrics": metrics,
        "comparison": comparison,
    }


def _verify_task_revision(
    *,
    examples: Sequence[TaskExample],
    baseline_metrics: Sequence[ExampleMetrics],
    sample_count: int,
    **execution_options: Any,
) -> dict[str, Any]:
    """Compare a deterministic, uniformly spaced subset before accepting a revision."""
    indices = _uniform_indices(len(examples), sample_count)
    records = [
        _verify_revision_execution(
            example=examples[index],
            baseline_metrics=baseline_metrics[index].to_dict(),
            enforce_improvement=False,
            **execution_options,
        )
        for index in indices
    ]
    baseline = summarize_metrics(baseline_metrics[index] for index in indices).to_dict()
    revised = {
        phase: {
            key: sum(record["metrics"][phase][key] for record in records) / len(records)
            for key in records[0]["metrics"][phase]
        }
        for phase in ("retrieval", "generation")
    }
    comparison = _compare_revision_metrics(
        selected_skill=execution_options["selected_skill"],
        baseline=baseline,
        revised=revised,
        prediction="",
        gold_answers=(),
        answer_supported=all(
            _prediction_contains_gold(record["prediction"], examples[index].answers)
            for index, record in zip(indices, records, strict=True)
        ),
    )
    if not comparison["improved"]:
        raise TaskWorkflowError(str(comparison["reason"]))
    return {
        "scope": "uniform_task_subset",
        "configured_examples": sample_count,
        "verified_examples": len(indices),
        "task_examples": len(examples),
        "example_indices": indices,
        "question_ids": [examples[index].id for index in indices],
        "baseline_summary": baseline,
        "revised_summary": revised,
        "comparison": comparison,
        "examples": records,
    }


def _uniform_indices(total: int, requested: int) -> list[int]:
    if total < 1 or requested < 1:
        raise TaskWorkflowError("Validation requires positive sample and task sizes")
    count = min(total, requested)
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def _compare_revision_metrics(
    *,
    selected_skill: str,
    baseline: Mapping[str, Any],
    revised: Mapping[str, Any],
    prediction: str,
    gold_answers: Sequence[str],
    answer_supported: bool | None = None,
) -> dict[str, Any]:
    """按 Skill 所属阶段比较核心指标，拒绝无提升或明显回退的修订。"""
    retrieval_keys = ("F1", "MRR", "Hit@10", "MAP", "NDCG")
    generation_keys = ("ChrF++", "METEOR", "R1", "RL")
    before_retrieval = _metric_average(baseline, "retrieval", retrieval_keys)
    after_retrieval = _metric_average(revised, "retrieval", retrieval_keys)
    before_generation = _metric_average(baseline, "generation", generation_keys)
    after_generation = _metric_average(revised, "generation", generation_keys)
    stage = _skill_metric_stage(selected_skill)
    tolerance = 1e-6

    if after_retrieval + tolerance < before_retrieval:
        reason = (
            "Post-revision retrieval score regressed: "
            f"{before_retrieval:.6f} -> {after_retrieval:.6f}"
        )
        improved = False
    elif stage == "retrieval":
        improved = after_retrieval > before_retrieval + tolerance
        reason = "Retrieval metrics did not measurably improve"
    elif stage == "generation":
        if answer_supported is None:
            answer_supported = _prediction_contains_gold(prediction, gold_answers)
        improved = (
            answer_supported
            and after_generation > before_generation + tolerance
        )
        reason = (
            "Generation metrics did not measurably improve with a gold-supported "
            "short answer"
        )
    else:
        retrieval_gain = after_retrieval - before_retrieval
        generation_gain = after_generation - before_generation
        improved = max(retrieval_gain, generation_gain) > tolerance
        reason = "The revision did not measurably improve retrieval or generation"

    return {
        "improved": improved,
        "stage": stage,
        "reason": None if improved else reason,
        "baseline_retrieval_score": before_retrieval,
        "revised_retrieval_score": after_retrieval,
        "baseline_generation_score": before_generation,
        "revised_generation_score": after_generation,
    }


def _metric_average(
    metrics: Mapping[str, Any],
    group: str,
    keys: Sequence[str],
) -> float:
    values = metrics.get(group, {})
    if not isinstance(values, Mapping):
        return 0.0
    numeric = [
        float(values[key])
        for key in keys
        if isinstance(values.get(key), (int, float))
    ]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _skill_metric_stage(selected_skill: str) -> str:
    normalized = selected_skill.lower()
    if "generator" in normalized:
        return "generation"
    if any(
        marker in normalized
        for marker in ("retriever", "reranker", "rewriter", "hyde", "bm25", "vector")
    ):
        return "retrieval"
    return "combined"


def _prediction_contains_gold(prediction: str, gold_answers: Sequence[str]) -> bool:
    normalized_prediction = _normalize_answer_text(prediction)
    return any(
        normalized_gold and normalized_gold in normalized_prediction
        for answer in gold_answers
        if (normalized_gold := _normalize_answer_text(answer))
    )


def _normalize_answer_text(value: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in value).split()
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析任务流配置、task 数量和单 task 样本上限。"""
    parser = argparse.ArgumentParser(
        description="Run a sequential multi-task RAG Skill optimization workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Task workflow YAML (default: self_optimization/task_flow.yaml).",
    )
    parser.add_argument("--max-tasks", type=_positive_cli_int, default=None)
    parser.add_argument("--examples-per-task", type=_positive_cli_int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--group", type=Path, default=None, help="Override the Group manifest or directory.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Override the Group corpus shuffle seed.")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration, data and selected tasks without model calls. With --group and no --config, validate data only.")
    args = parser.parse_args(argv)
    if args.validate_only and args.run_id is not None:
        parser.error("--run-id only applies to execution, not --validate-only")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """加载任务流配置并顺序执行全部 RAG task。"""
    configure_console_utf8()
    args = parse_args(argv)
    if args.validate_only and args.group is not None and args.config is None:
        group = load_group(
            args.group, shuffle_seed=42 if args.shuffle_seed is None else args.shuffle_seed
        )
        tasks = tuple(RAGTaskConfig(task.task_id, group.dataset, task.path) for task in group.tasks)
        selected = _select_task_examples(
            tasks, group=group, max_tasks=args.max_tasks,
            examples_per_task=args.examples_per_task,
        )
        summary = {
            **_group_summary(group),
            "validation_scope": "group_data",
            "selection": _selection_summary(selected, args.max_tasks, args.examples_per_task),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    config = load_task_workflow_config(
        args.config or Path(__file__).with_name("task_flow.yaml"),
        group_path=args.group, shuffle_seed=args.shuffle_seed,
    )
    if args.validate_only:
        summary = validate_task_workflow(
            config, max_tasks=args.max_tasks, examples_per_task=args.examples_per_task
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    run_task_workflow(
        config,
        max_tasks=args.max_tasks,
        examples_per_task=args.examples_per_task,
        run_id=args.run_id,
    )


def _positive_cli_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def validate_task_workflow(
    config: TaskWorkflowConfig,
    *,
    max_tasks: int | None = None,
    examples_per_task: int | None = None,
) -> dict[str, Any]:
    """Inspect exactly the tasks and QA that execution selects, without creating clients."""
    if not isinstance(config, TaskWorkflowConfig):
        raise TaskWorkflowError("Expected TaskWorkflowConfig")
    selected = _select_task_examples(
        config.tasks, group=config.group, max_tasks=max_tasks,
        examples_per_task=examples_per_task,
    )
    return {
        "validation_scope": "configuration_and_data",
        "task_flow_config": str(config.config_path),
        "group": _group_summary(config.group) if config.group is not None else None,
        "selection": _selection_summary(selected, max_tasks, examples_per_task),
        "validation_examples_per_task": config.validation_examples_per_task,
    }


def _select_task_examples(
    tasks: Sequence[RAGTaskConfig],
    *,
    group: DataGroup | None = None,
    max_tasks: int | None = None,
    examples_per_task: int | None = None,
) -> tuple[tuple[RAGTaskConfig, tuple[TaskExample, ...]], ...]:
    """Use one selection contract for validation, execution and saved reports."""
    _validate_limit(max_tasks, "max_tasks")
    _validate_limit(examples_per_task, "examples_per_task")
    selected_tasks = tasks if max_tasks is None else tasks[:max_tasks]
    if not selected_tasks:
        raise TaskWorkflowError("No tasks were selected")
    if len({task.name for task in selected_tasks}) != len(selected_tasks):
        raise TaskWorkflowError("Selected task IDs must be unique")
    selected = []
    for task in selected_tasks:
        _validate_limit(task.max_examples, f"Task {task.name!r} max_examples")
        examples = (
            _group_task_examples(group, task, group.documents)
            if group is not None else _load_task_examples(task.data_path)
        )
        limit = _effective_task_limit(task, examples_per_task)
        if limit is not None:
            examples = examples[:limit]
        if not examples:
            raise TaskWorkflowError(f"Task {task.name!r} contains no selected examples")
        unexpected = sorted({example.dataset for example in examples if example.dataset != task.dataset})
        if unexpected:
            raise TaskWorkflowError(
                f"Task {task.name!r} expects dataset {task.dataset!r}, but records use: {unexpected}"
            )
        selected.append((task, tuple(examples)))
    return tuple(selected)


def _selection_summary(
    selected: Sequence[tuple[RAGTaskConfig, Sequence[TaskExample]]],
    max_tasks: int | None,
    examples_per_task: int | None,
) -> dict[str, Any]:
    return {
        "max_tasks": max_tasks,
        "examples_per_task": examples_per_task,
        "task_count": len(selected),
        "qa_count": sum(len(examples) for _, examples in selected),
        "tasks": [
            {
                "task_id": task.name,
                "group_id": examples[0].group_id,
                "dataset": task.dataset,
                "qa_count": len(examples),
                "qa_ids": [example.id for example in examples],
            }
            for task, examples in selected
        ],
    }


def _group_summary(group: DataGroup) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "dataset": group.dataset,
        "manifest": str(group.manifest_path),
        "corpus_fingerprint": group.corpus_fingerprint,
        "document_count": len(group.documents),
        "task_count": len(group.tasks),
        "qa_count": group.qa_count,
        "shuffle_seed": group.shuffle_seed,
    }


def _group_task_examples(
    group: DataGroup,
    task: RAGTaskConfig,
    documents: tuple[Mapping[str, Any], ...],
) -> list[TaskExample]:
    matched = next((item for item in group.tasks if item.path == task.data_path.resolve()), None)
    if matched is None or matched.task_id != task.name or group.dataset != task.dataset:
        raise TaskWorkflowError(f"Task {task.name!r} does not belong to the loaded Group")
    return [
        TaskExample(
            id=example.id,
            dataset=group.dataset,
            question=example.question,
            answers=example.answers,
            documents=documents,
            relevant_document_ids=example.relevant_document_ids,
            label_type=example.label_type,
            group_id=group.group_id,
            task_id=matched.task_id,
        )
        for example in matched.examples
    ]


def _execution_request(
    framework_config: Any, task: RAGTaskConfig, example: TaskExample
) -> dict[str, Any]:
    """Keep evaluation annotations out of the Executor and reuse the shared corpus."""
    options = {**framework_config.request_defaults, **task.request}
    return build_rag_request(example.question, example.documents, options=options)


def _load_task_examples(path: Path) -> list[TaskExample]:
    """读取规范 task JSONL 并验证答案、文档与证据 ID。"""
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskWorkflowError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise TaskWorkflowError(f"Task record at {path}:{line_number} must be an object")
        documents_value = value.get("documents")
        if isinstance(documents_value, (str, bytes, bytearray)) or not isinstance(
            documents_value,
            Sequence,
        ):
            raise TaskWorkflowError(f"Task record at {path}:{line_number} has no documents")
        documents = tuple(_normalize_document(item, path, line_number) for item in documents_value)
        document_ids = [str(document["id"]) for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise TaskWorkflowError(f"Duplicate document IDs at {path}:{line_number}")
        answers = _text_sequence(value, "answers", path, line_number)
        relevant = _text_sequence(
            value,
            "relevant_document_ids",
            path,
            line_number,
        )
        missing = set(relevant) - set(document_ids)
        if missing:
            raise TaskWorkflowError(
                f"Relevant documents are missing at {path}:{line_number}: {sorted(missing)}"
            )
        label_type_value = value.get("label_type")
        label_type = None if label_type_value is None else str(label_type_value)
        records.append(
            TaskExample(
                id=_record_text(value, "id", path, line_number),
                dataset=_record_text(value, "dataset", path, line_number),
                question=_record_text(value, "question", path, line_number),
                answers=answers,
                documents=documents,
                relevant_document_ids=relevant,
                label_type=label_type,
            )
        )
    return records


def _normalize_document(
    value: Any,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    """校验 task 候选文档的 ID、正文和可选标题。"""
    if not isinstance(value, Mapping):
        raise TaskWorkflowError(f"Invalid document at {path}:{line_number}")
    document_id = _record_text(value, "id", path, line_number)
    text = value.get("text")
    if not isinstance(text, str):
        raise TaskWorkflowError(f"Document '{document_id}' has no text")
    document = dict(value)
    document["id"] = document_id
    document["text"] = text
    return document


def _record_text(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    """读取 task 记录中的必需非空字符串。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskWorkflowError(f"'{key}' is invalid at {path}:{line_number}")
    return value.strip()


def _text_sequence(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    """读取 task 记录中的非空字符串列表。"""
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TaskWorkflowError(f"'{key}' is invalid at {path}:{line_number}")
    normalized = tuple(str(item).strip() for item in value)
    if not normalized or any(not item for item in normalized):
        raise TaskWorkflowError(f"'{key}' is empty at {path}:{line_number}")
    return normalized


def _effective_task_limit(
    task: RAGTaskConfig,
    examples_per_task: int | None,
) -> int | None:
    """合并 task 自身上限与命令行临时上限并取更严格值。"""
    limits = [value for value in (task.max_examples, examples_per_task) if value is not None]
    return min(limits) if limits else None


def _validate_limit(value: int | None, name: str) -> None:
    """验证命令行可选上限为正整数或 null。"""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise TaskWorkflowError(f"{name} must be positive or null")


def _build_workflow_report(
    *,
    config: TaskWorkflowConfig,
    workspace: SkillWorkspace,
    tasks: Sequence[Mapping[str, Any]],
    overall_summary: Mapping[str, Any],
    event_log: OptimizationEventLogger,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """构造不包含密钥和完整候选文档的 task workflow 报告。"""
    return {
        "schema_version": 1,
        "run_id": workspace.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "task_flow_config": str(config.config_path),
        "group": _group_summary(config.group) if config.group is not None else None,
        "selection": dict(selection),
        "validation_examples_per_task": config.validation_examples_per_task,
        "source_skill_root": str(workspace.source_root),
        "source_skill_hash": workspace.source_hash,
        "working_skill_root": str(workspace.skill_root),
        "optimizer": {
            "provider": config.self_optimization.optimizer.provider,
            "model": config.self_optimization.optimizer.model,
            "base_url": config.self_optimization.optimizer.base_url,
        },
        "completed_tasks": len(tasks),
        "overall_summary": dict(overall_summary),
        "tasks": [dict(task) for task in tasks],
        "artifacts": {
            "report_path": str(workspace.run_root / "task-flow.report.json"),
            "log_path": str(event_log.path),
            "revisions_root": str(workspace.revisions_root),
        },
    }


if __name__ == "__main__":
    main()
