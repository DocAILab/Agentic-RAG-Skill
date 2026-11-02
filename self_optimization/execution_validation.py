"""Operational revision checks. Answers and evaluation scores never gate retention."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from framework import ModelClient, RuntimeComponentContext, select_rag_plan
from framework.compiler import ExecutionError, run_compiled_rag


def participating_names(selection: Mapping[str, Any]) -> tuple[str, ...]:
    """List participating packages in stable Manage, Agentic and binding order."""
    names = [selection.get("manage_skill"), selection.get("agentic_skill")]
    for values in selection.get("component_bindings", {}).values():
        names.extend(values)
    return tuple(dict.fromkeys(name for name in names if isinstance(name, str)))


def prepare_validation_selection(
    *, selection: Mapping[str, Any], selected_skill: str,
    task_selection_request: Mapping[str, Any], skill_root: Path,
    manage_skill: str, executor_model: ModelClient,
) -> tuple[Mapping[str, Any], str]:
    """Plan one validation route for the Task before running its smoke QA.

    Manage instructions control selection, so a Manage revision must exercise
    the complete Task routing request once. Other revisions keep the Task's
    original bindings to avoid selecting around the package being checked.
    """
    if selected_skill != selection.get("manage_skill"):
        return selection, "reused_task_route"
    questions = task_selection_request.get("questions")
    if (
        task_selection_request.get("selection_scope") != "task_shared"
        or not isinstance(questions, list)
        or not questions
    ):
        raise ExecutionError("Manage validation requires the complete shared Task routing request")
    plan = select_rag_plan(
        task_selection_request, model=executor_model, skill_root=skill_root,
        manage_skill=manage_skill,
    )
    return plan.to_dict(), "reselected_task_route"


def verify_execution(
    *, request: Mapping[str, Any], selection: Mapping[str, Any],
    selected_skill: str, skill_root: Path,
    runtime_context: RuntimeComponentContext, selection_mode: str,
) -> dict[str, Any]:
    """Execute one QA under the prepared Task route without further selection."""
    if selection_mode not in {"reselected_task_route", "reused_task_route"}:
        raise ExecutionError("Unknown operational validation selection mode")
    if selected_skill not in participating_names(selection):
        raise ExecutionError("Execution check did not bind the revised Skill")
    result = run_compiled_rag(
        workflow=selection["agentic_skill"], bindings=selection["component_bindings"],
        request=request, skill_root=skill_root, context=runtime_context,
    )
    exercised = True
    if selected_skill not in {selection["manage_skill"], selection["agentic_skill"]}:
        exercised = any(t.get("component") == selected_skill
                        for t in result.get("component_timings", []))
    return {
        "prediction": result["answer"].strip(),
        "agentic_skill": selection["agentic_skill"],
        "retrieved_document_ids": [d["id"] for d in result["documents"]],
        "validation_status": "passed", "selected_skill_exercised": exercised,
        "coverage_status": "exercised" if exercised else "not_exercised",
        "selection_mode": selection_mode,
    }
