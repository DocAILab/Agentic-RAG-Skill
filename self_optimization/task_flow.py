"""在同一个 Skill 工作副本上顺序执行多个 RAG task 并逐 task 优化。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from framework.data_groups import DataGroup, load_group
from framework.selection import build_task_selection_request

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
from .checkpoint import (TaskBoundaryExit, TaskCheckpoint, exclusive_workflow,
                         restore_metrics, workflow_identity)
from .execution_validation import (
    participating_names,
    prepare_validation_selection,
    verify_execution,
)
from .console import configure_console_utf8
from .optimizer import OptimizationError, OptimizationNoChange, SkillOptimizer
from .runtime import (
    OptimizationEventLogger,
    _build_retrieval_process,
    _retrieved_ids,
    _safe_error_message,
    _write_json,
    task_retrieval_process,
)
from .task_config import (
    RAGTaskConfig,
    TaskWorkflowConfig,
    load_task_workflow_config,
)
from .telemetry import ModelCaptureRecorder, capture_context
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


@exclusive_workflow
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
    resume: bool = False,
    freeze_skills: bool = False,
    task_budget_seconds: float | None = None,
    selection_scope: str = "task_shared",
    capture_model_io: bool = True,
) -> dict[str, Any]:
    """依次运行所有 task，并在每个 task 汇总评估后优化一次 Skill。"""
    if not isinstance(config, TaskWorkflowConfig):
        raise TaskWorkflowError(
            "Expected TaskWorkflowConfig"
        )
    started_at = time.monotonic()
    if task_budget_seconds is not None and (
        not math.isfinite(task_budget_seconds) or task_budget_seconds <= 0
    ):
        raise TaskWorkflowError("task_budget_seconds must be finite and positive")
    if selection_scope != "task_shared":
        raise TaskWorkflowError("This experiment requires selection_scope=task_shared")
    selected = _select_task_examples(
        config.tasks, group=config.group, max_tasks=max_tasks,
        examples_per_task=examples_per_task,
    )
    selection = _selection_summary(selected, max_tasks, examples_per_task)
    selection["selection_scope"] = selection_scope
    base = config.self_optimization
    framework_config = base.framework
    workspace_factory = SkillWorkspace.open_existing if resume else SkillWorkspace.create
    workspace = workspace_factory(
        framework_config.skill_root,
        base.runs_root,
        run_id=run_id or base.run_id,
    )
    event_log = OptimizationEventLogger(
        workspace.run_root / "task-flow.events.jsonl",
        run_id=workspace.run_id,
    )
    checkpoint = TaskCheckpoint(workspace, workflow_identity(config, selection, freeze_skills), resume=resume)
    checkpoint.recover_events(event_log)
    event_log.write(
        "workflow_started",
        task_flow_config=str(config.config_path),
        source_skill_root=str(workspace.source_root),
        working_skill_root=str(workspace.skill_root),
        group_id=config.group.group_id,
        task_names=[task.name for task, _ in selected],
        selection=selection,
        selection_scope=selection_scope,
        optimizer_model=None if freeze_skills else base.optimizer.model,
        optimizer_mode=None if freeze_skills else base.policy_mode,
        experiment_arm="frozen_baseline" if freeze_skills else "adaptive",
        optimization_enabled=not freeze_skills,
        revision_validation_enabled=not freeze_skills and base.optimization.verify_revision_execution,
        resumed=resume,
        previously_completed_tasks=checkpoint.completed_tasks,
        revision_validation_mode="execution_only",
        capture_model_io=capture_model_io,
    )

    if executor_model is None:
        executor_model, configured_embedding = create_clients_from_config(
            framework_config
        )
        if embedding_model is None:
            embedding_model = configured_embedding
    if not freeze_skills and optimizer_model is None:
        optimizer_model = create_optimizer_model(base)
    recorder = None
    if capture_model_io:
        recorder = ModelCaptureRecorder(
            workspace.run_root, source_skill_root=workspace.source_root,
            skill_hash=workspace.working_hash, source_hash=workspace.source_hash,
            corpus_fingerprint=config.group.corpus_fingerprint,
            warning_callback=lambda warning: event_log.write("model_capture_warning", **warning),
        )
        executor_model = recorder.wrap(executor_model, "executor")
        if optimizer_model is not None and not freeze_skills:
            optimizer_model = recorder.wrap(optimizer_model, "policy")
    optimizer = None if freeze_skills else SkillOptimizer(optimizer_model, base.optimization)
    runtime_context = RuntimeComponentContext(
        executor_model=executor_model,
        component_observer=recorder,
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
    shared_documents = runtime_context.bind_corpus(
        config.group.documents, config.group.corpus_fingerprint
    )
    if shared_documents is not config.group.documents:
        selected = tuple(
            (task, tuple(replace(example, documents=shared_documents) for example in examples))
            for task, examples in selected
        )

    report_path = workspace.run_root / "task-flow.report.json"
    task_records = list((checkpoint.completed_report or {}).get("tasks", []))
    overall_metrics: list[ExampleMetrics] = restore_metrics(task_records)
    for task_index, (task, task_examples) in enumerate(selected, start=1):
        if task_index <= checkpoint.completed_tasks:
            continue
        if task_budget_seconds is not None and time.monotonic() - started_at >= task_budget_seconds:
            workspace.assert_source_unchanged()
            event_log.write("workflow_paused_at_task_boundary", completed_tasks=len(task_records),
                            next_task_id=task.name, working_skill_hash=workspace.working_hash(),
                            elapsed_seconds=time.monotonic() - started_at, exit_code=75)
            raise TaskBoundaryExit()
        attempt_id = checkpoint.begin_task(task.name, task_index)
        task_identity = {
            "task_id": task.name,
            "group_id": config.group.group_id,
            "attempt_id": attempt_id,
        }
        if recorder is not None:
            recorder.update_context(**task_identity, scope="primary_task_routing", question_id=None)
        event_log.write(
            "task_started",
            **task_identity,
            task_index=task_index,
            total_tasks=len(selected),
            task_name=task.name,
            dataset=task.dataset,
            examples=len(task_examples),
            skill_hash_before=workspace.working_hash(),
            selection_scope=selection_scope,
        )
        if verbose:
            print(f"\n=== Task {task_index}/{len(selected)}: {task.name} ===")

        task_requests = [
            _execution_request(framework_config, task, example)
            for example in task_examples
        ]
        routing_request = build_task_selection_request(
            task_requests, task_id=task.name, dataset=task.dataset,
            question_ids=[example.id for example in task_examples],
        )
        routing_directory = workspace.run_root / "task_routes" / f"task-{task_index:04d}" / attempt_id
        request_path = routing_directory / "request.json"
        plan_path = routing_directory / "plan.json"
        _write_json(request_path, routing_request)
        routing_request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
        event_log.write(
            "task_routing_started", **task_identity, selection_scope=selection_scope,
            question_count=len(task_examples), request_path=str(request_path),
            request_sha256=routing_request_sha256, skill_hash=workspace.working_hash(),
        )
        try:
            plan = select_rag_plan(
                routing_request, model=executor_model, skill_root=workspace.skill_root,
                manage_skill=framework_config.manage_skill,
            )
        except Exception as exc:
            _record_task_execution_failure(
                exc, stage="task_routing", workspace=workspace, event_log=event_log,
                checkpoint=checkpoint, task_identity=task_identity, task_index=task_index,
                completed_examples=0, selection_scope=selection_scope,
            )
            raise
        plan_payload = plan.to_dict()
        task_plan_hash = hashlib.sha256(json.dumps(
            plan_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        _write_json(plan_path, plan_payload)
        task_routing = {
            "selection_scope": selection_scope,
            "task_plan_hash": task_plan_hash,
            "question_count": len(task_examples),
            "request_path": str(request_path),
            "request_sha256": routing_request_sha256,
            "plan_path": str(plan_path),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "skill_hash_at_selection": workspace.working_hash(),
            "decision_model": getattr(executor_model, "model", None),
            "decision_rounds": 1,
            "expected_selection_calls_without_repairs": 3,
            "plan": plan_payload,
        }
        event_log.write("task_routing_completed", **task_identity, **task_routing)
        executions = []
        candidate_names = []
        for example_index, (example, request) in enumerate(zip(task_examples, task_requests, strict=True), start=1):
            if recorder is not None:
                recorder.update_context(scope="primary_qa", question_id=example.id)
            event_log.write(
                "task_example_started",
                **task_identity,
                task_name=task.name,
                example_index=example_index,
                total_examples=len(task_examples),
                question_id=example.id,
                label_type=example.label_type,
                question=example.question,
                selection_scope=selection_scope,
                task_plan_hash=task_plan_hash,
            )
            try:
                command = compile_rag_command(
                    plan,
                    skill_root=workspace.skill_root,
                    context=runtime_context,
                )
                if recorder is not None:
                    recorder.record_compile(command)
                result = command.run(request)
            except Exception as exc:
                _record_task_execution_failure(
                    exc, stage="qa_execution", workspace=workspace, event_log=event_log,
                    checkpoint=checkpoint, task_identity=task_identity, task_index=task_index,
                    completed_examples=len(executions), selection_scope=selection_scope,
                    question_id=example.id, example_index=example_index,
                    task_plan_hash=task_plan_hash,
                )
                raise
            result["selection"] = plan.to_dict()
            result["compiled_instruction"] = command.instruction
            candidate_names.extend(participating_names(plan.to_dict()))
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
                selection_scope=selection_scope,
                task_plan_hash=task_plan_hash,
            )
            if verbose:
                print(
                    f"[{example_index}/{len(task_examples)}] {example.id}: "
                    f"{result.get('answer')}"
                )

        task_metrics = []
        example_records = []
        process_examples = []
        process_indices = set(_uniform_indices(len(executions), config.max_process_examples))
        for execution_index, (example, result) in enumerate(executions):
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
                    "selection_scope": selection_scope,
                    "task_plan_hash": task_plan_hash,
                }
            )
            if execution_index in process_indices:
                process_examples.append(
                    {"question_id": example.id, "example_index": execution_index + 1,
                     **_build_retrieval_process(
                        question=example.question,
                        result=result,
                        max_documents=base.optimization.max_documents,
                        excerpt_chars=base.optimization.document_excerpt_chars,
                    )}
                )

        task_summary = summarize_metrics(task_metrics).to_dict()
        overall_summary = summarize_metrics(overall_metrics).to_dict()
        task_evaluation = {
            "task_name": task.name,
            "selection_scope": selection_scope,
            "task_plan_hash": task_plan_hash,
            "task_id": task.name,
            "attempt_id": attempt_id,
            "group_id": config.group.group_id,
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
        if freeze_skills:
            optimization_record = {
                "status": "disabled_baseline",
                "reason": "Frozen original Skills; Policy and revision validation are disabled.",
            }
            if workspace.working_hash() != workspace.source_hash:
                raise TaskWorkflowError("Frozen baseline Skills changed during execution")
            event_log.write("task_optimization_disabled_baseline", **task_identity, **optimization_record)
        else:
            if recorder is not None:
                recorder.update_context(scope="optimization", question_id=None)
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
                    retrieval_process=task_retrieval_process(
                        task_name=task.name, dataset=task.dataset,
                        executed_examples=len(executions), examples=process_examples,
                        selection=plan_payload, task_plan_hash=task_plan_hash,
                        max_chars=base.optimization.max_process_chars,
                    ),
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
                            task_selection_request=routing_request,
                            baseline_selections=[result["selection"] for _, result in executions],
                            baseline_timings=[result.get("component_timings", []) for _, result in executions],
                            sample_count=config.validation_examples_per_task,
                            task=task,
                            selected_skill=proposal.selected_skill,
                            framework_config=framework_config,
                            workspace=workspace,
                            executor_model=executor_model,
                            runtime_context=runtime_context,
                        )
                    except Exception as exc:
                        category = (
                            "external_service_error" if isinstance(exc, ModelAPIError)
                            else "execution_validation_failed"
                        )
                        reason = "Post-revision operational validation failed: " + _safe_error_message(exc)
                        workspace.rollback_revision(
                            revision, reason=reason, failure_category=category,
                            verification=getattr(exc, "revision_verification", None),
                        )
                        if isinstance(exc, ModelAPIError):
                            exc.failure_category = category
                            raise
                        rejected = RevisionRejected(reason)
                        rejected.failure_category = category
                        rejected.revision_verification = getattr(exc, "revision_verification", None)
                        raise rejected from exc
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
                    "failure_category": getattr(exc, "failure_category", "external_service_error" if isinstance(exc, ModelAPIError) else "proposal_validation_failed"),
                }
                if isinstance(exc, ModelAPIError) and callable(getattr(exc, "diagnostics", None)):
                    optimization_record["model_error"] = exc.diagnostics()
                if getattr(exc, "revision_verification", None) is not None:
                    optimization_record["verification"] = exc.revision_verification
            audit = write_optimization_audit(
                workspace.run_root / "optimizer_calls" / f"task-{task_index:04d}" / attempt_id,
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
            "group_id": config.group.group_id,
            "dataset": task.dataset,
            "data_path": str(task.data_path),
            "summary": task_summary,
            "overall_summary_after_task": overall_summary,
            "candidate_skills": list(unique_candidates),
            "selection_scope": selection_scope,
            "task_plan_hash": task_plan_hash,
            "task_routing": task_routing,
            "examples": example_records,
            "optimization": optimization_record,
            "attempt_id": attempt_id,
            "skill_hash_before": checkpoint.state["active_task"]["skill_hash"],
            "skill_hash_after": workspace.working_hash(),
        }
        task_records.append(task_record)
        checkpoint.commit_task(
            _build_workflow_report(
                config=config,
                workspace=workspace,
                tasks=task_records,
                overall_summary=overall_summary,
                event_log=event_log,
                selection=selection,
                freeze_skills=freeze_skills,
            ),
            event_log,
            {**task_identity, "task_name": task.name, "summary": task_summary,
             "optimization_status": optimization_record["status"], "skill_hash_after": workspace.working_hash()},
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
        freeze_skills=freeze_skills,
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


def _record_task_execution_failure(
    error: Exception,
    *,
    stage: str,
    workspace: SkillWorkspace,
    event_log: OptimizationEventLogger,
    checkpoint: TaskCheckpoint,
    task_identity: Mapping[str, Any],
    task_index: int,
    completed_examples: int,
    selection_scope: str,
    question_id: str | None = None,
    example_index: int | None = None,
    task_plan_hash: str | None = None,
) -> None:
    """Preserve partial-attempt evidence without committing it or changing retries."""
    path = (workspace.run_root / "task_failures" / f"task-{task_index:04d}"
            / task_identity["attempt_id"] / "failure.json")
    record = {
        **task_identity,
        "task_index": task_index,
        "stage": stage,
        "question_id": question_id,
        "example_index": example_index,
        "selection_scope": selection_scope,
        "task_plan_hash": task_plan_hash,
        "committed": False,
        "committed_tasks_before_attempt": checkpoint.completed_tasks,
        "uncommitted_executed_examples": completed_examples,
        "partial_results_path": str(event_log.path),
        "diagnostic_path": str(path),
        "error_type": type(error).__name__,
        # Arbitrary component exceptions may contain credentials or request text.
        # Only the ModelAPIError's allowlisted diagnostics may expose provider detail.
        "failure_category": "external_service_error" if isinstance(error, ModelAPIError)
        else "task_execution_error",
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    try:
        if isinstance(error, ModelAPIError) and callable(getattr(error, "diagnostics", None)):
            record["model_error"] = error.diagnostics()
        _write_json(path, record)
        event_log.write("task_execution_failed", **record)
    except Exception as evidence_error:
        # A full disk or damaged output directory must not replace the original
        # execution exception or make the active Task look committed.
        error.add_note("Could not persist Task failure evidence: " + type(evidence_error).__name__)


def _verify_revision_execution(
    *, example: TaskExample, task: RAGTaskConfig, selected_skill: str,
    framework_config: Any, workspace: SkillWorkspace,
    runtime_context: RuntimeComponentContext,
    selection: Mapping[str, Any], selection_mode: str,
) -> dict[str, Any]:
    """Exercise revised code and interfaces; never score its answer quality."""
    with capture_context(runtime_context.executor_model, scope="revision_validation", question_id=example.id):
        record = verify_execution(
            request=_execution_request(framework_config, task, example),
            selection=selection, selected_skill=selected_skill,
            skill_root=workspace.skill_root, runtime_context=runtime_context,
            selection_mode=selection_mode,
        )
    return {"question_id": example.id, "group_id": example.group_id,
            "task_id": example.task_id or task.name, **record}


def _verify_task_revision(
    *, examples: Sequence[TaskExample], sample_count: int,
    baseline_selections: Sequence[Mapping[str, Any]],
    baseline_timings: Sequence[Sequence[Mapping[str, Any]]],
    executor_model: ModelClient,
    task_selection_request: Mapping[str, Any], **execution_options: Any,
) -> dict[str, Any]:
    """Smoke-test participating QA under one shared Task validation plan."""
    selected_skill = execution_options["selected_skill"]
    eligible = [i for i, selection in enumerate(baseline_selections)
                if selected_skill in participating_names(selection)]
    exercised = [i for i in eligible if selected_skill in {
        baseline_selections[i].get("manage_skill"),
        baseline_selections[i].get("agentic_skill"),
    } or any(t.get("component") == selected_skill for t in baseline_timings[i])]
    if exercised:
        eligible = exercised
    if not eligible:
        raise TaskWorkflowError("No Task route exercises the revised Skill")
    indices = [eligible[i] for i in _uniform_indices(len(eligible), sample_count)]
    verification = {
        "mode": "execution_only", "performance_gate": False,
        "scope": "participating_task_subset", "configured_examples": sample_count,
        "verified_examples": 0, "task_examples": len(examples),
        "example_indices": indices, "question_ids": [examples[i].id for i in indices],
        "all_passed": False, "examples": [],
    }
    try:
        with capture_context(executor_model, scope="revision_validation", question_id=None):
            validation_selection, selection_mode = prepare_validation_selection(
                selection=baseline_selections[indices[0]], selected_skill=selected_skill,
                task_selection_request=task_selection_request,
                skill_root=execution_options["workspace"].skill_root,
                manage_skill=execution_options["framework_config"].manage_skill,
                executor_model=executor_model,
            )
    except Exception as exc:
        verification["planning_error"] = {
            "error_type": type(exc).__name__, "error": _safe_error_message(exc),
        }
        exc.revision_verification = verification
        raise
    verification["selection_mode"] = selection_mode
    verification["plan"] = dict(validation_selection)
    for index in indices:
        try:
            record = _verify_revision_execution(
                example=examples[index],
                selection=validation_selection, selection_mode=selection_mode,
                **execution_options,
            )
        except Exception as exc:
            verification["examples"].append({
                "question_id": examples[index].id, "validation_status": "failed",
                "error_type": type(exc).__name__, "error": _safe_error_message(exc),
            })
            exc.revision_verification = verification
            raise
        verification["examples"].append(record)
        verification["verified_examples"] += 1
    verification["all_passed"] = True
    verification["exercised_examples"] = sum(
        record.get("selected_skill_exercised") is True for record in verification["examples"]
    )
    exercised_count = verification["exercised_examples"]
    verification["coverage_status"] = (
        "full" if exercised_count == len(indices) else "partial" if exercised_count else "none"
    )
    return verification


def _uniform_indices(total: int, requested: int) -> list[int]:
    if total < 1 or requested < 1:
        raise TaskWorkflowError("Validation requires positive sample and task sizes")
    count = min(total, requested)
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


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
    parser.add_argument("--resume", action="store_true", help="Resume the same committed run-id/workspace; never creates a fresh baseline.")
    parser.add_argument("--freeze-skills", action="store_true", help="Run the original Skills without constructing or calling the Policy model.")
    parser.add_argument("--no-capture-model-io", action="store_true",
                        help="Disable the default redacted prompt/response observations and source snapshots.")
    parser.add_argument("--policy-mode", choices=("local", "api"), default=None,
                        help="Select the local or API Policy profile; no automatic fallback.")
    parser.add_argument("--selection-scope", choices=("task_shared",), default="task_shared",
                        help="Select one RAG plan from all questions per Task, then reuse its route and bindings.")
    parser.add_argument("--task-budget-seconds", type=float, default=None,
                        help="At the next Task boundary after this elapsed budget, checkpoint and exit 75.")
    parser.add_argument("--group", type=Path, default=None, help="Override the Group manifest or directory.")
    parser.add_argument("--shuffle-seed", type=int, default=None, help="Override the Group corpus shuffle seed.")
    parser.add_argument("--validate-only", action="store_true", help="Validate configuration, data and selected tasks without model calls. With --group and no --config, validate data only.")
    args = parser.parse_args(argv)
    if args.validate_only and args.run_id is not None:
        parser.error("--run-id only applies to execution, not --validate-only")
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")
    if args.task_budget_seconds is not None and (
        not math.isfinite(args.task_budget_seconds) or args.task_budget_seconds <= 0
    ):
        parser.error("--task-budget-seconds must be finite and positive")
    if args.validate_only and (args.resume or args.task_budget_seconds is not None):
        parser.error("--resume/--task-budget-seconds only apply to execution")
    if args.validate_only and args.policy_mode is not None and args.config is None:
        parser.error("--policy-mode with --validate-only requires --config")
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
        policy_mode=args.policy_mode,
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
        resume=args.resume,
        freeze_skills=args.freeze_skills,
        task_budget_seconds=args.task_budget_seconds,
        selection_scope=args.selection_scope,
        capture_model_io=not args.no_capture_model_io,
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
        "optimizer_mode": config.self_optimization.policy_mode,
        "task_flow_config": str(config.config_path),
        "group": _group_summary(config.group),
        "selection": _selection_summary(selected, max_tasks, examples_per_task),
        "validation_examples_per_task": config.validation_examples_per_task,
        "revision_validation_mode": "execution_only",
        "acceptance_policy": "execution_only",
    }


def _select_task_examples(
    tasks: Sequence[RAGTaskConfig],
    *,
    group: DataGroup,
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
        examples = _group_task_examples(group, task, group.documents)
        limit = examples_per_task
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
        "selection_scope": "task_shared",
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
    freeze_skills: bool = False,
) -> dict[str, Any]:
    """构造不包含密钥和完整候选文档的 task workflow 报告。"""
    return {
        "schema_version": 1,
        "run_id": workspace.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "task_flow_config": str(config.config_path),
        "group": _group_summary(config.group),
        "selection": dict(selection),
        "selection_scope": selection.get("selection_scope", "task_shared"),
        "validation_examples_per_task": config.validation_examples_per_task,
        "revision_validation_mode": "execution_only",
        "acceptance_policy": "execution_only",
        "experiment_arm": "frozen_baseline" if freeze_skills else "adaptive",
        "optimization_enabled": not freeze_skills,
        "revision_validation_enabled": not freeze_skills and config.self_optimization.optimization.verify_revision_execution,
        "source_skill_root": str(workspace.source_root),
        "source_skill_hash": workspace.source_hash,
        "working_skill_root": str(workspace.skill_root),
        "optimizer": None if freeze_skills else {
            "mode": config.self_optimization.policy_mode,
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
