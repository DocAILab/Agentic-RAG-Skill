"""Operational revision checks. Answers and evaluation scores never gate retention."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from framework import ModelClient, RuntimeComponentContext, select_rag_plan
from framework.compiler import ExecutionError, run_compiled_rag


def participating_names(selection: Mapping[str, Any]) -> set[str]:
    names = {selection.get("manage_skill"), selection.get("agentic_skill")}
    for values in selection.get("component_bindings", {}).values():
        names.update(values)
    return {name for name in names if isinstance(name, str)}


def verify_execution(
    *, request: Mapping[str, Any], selection: Mapping[str, Any] | None,
    selected_skill: str, skill_root: Path, manage_skill: str,
    executor_model: ModelClient, runtime_context: RuntimeComponentContext,
) -> dict[str, Any]:
    # A Manage edit needs fresh selection. Other edits reuse a route which
    # actually included the revised package, rather than selecting around it.
    reselect = selection is None or selected_skill == selection.get("manage_skill")
    if reselect:
        plan = select_rag_plan(request, model=executor_model, skill_root=skill_root,
                               manage_skill=manage_skill)
        selection = plan.to_dict()
    assert selection is not None
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
        "selection_mode": "reselected" if reselect else "reused_task_route",
    }
