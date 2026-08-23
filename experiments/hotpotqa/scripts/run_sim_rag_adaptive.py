"""Run HotpotQA with fixed SIM-RAG and adaptive Component selection."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.hotpotqa.scripts.run_sim_rag import (
    _load_jsonl,
)
from experiments.hotpotqa.scripts.sim_rag_adaptive_report import (
    evaluate_result,
    failure_record,
    success_record,
    write_checkpoint,
)
from framework import (
    AgenticStageResult,
    RAGSelectionPlan,
    RuntimeComponentContext,
    compile_rag_command,
    create_clients_from_config,
    discover_specs,
    load_framework_config,
    select_component_skills,
)

AGENTIC_SKILL = "agentic-sim-rag"


def run_adaptive_experiment(
    config,
    *,
    model=None,
    embedding_model=None,
    verbose=True,
):
    """Select Components per example, execute SIM-RAG, and write a report."""
    demo = config.demo
    if demo is None:
        raise ValueError("Adaptive experiment config requires a demo section")
    if model is None:
        model, configured_embedding = create_clients_from_config(config)
        if embedding_model is None:
            embedding_model = configured_embedding

    agentic_result = _fixed_agentic_result(config.skill_root)
    tests = _load_jsonl(demo.test_path)[: demo.max_examples]
    corpus = {item["id"]: item for item in _load_jsonl(demo.corpus_path)}
    outputs, evaluations, failures = [], [], []
    for index, example in enumerate(tests, start=1):
        output, evaluation, failure = _run_example(
            config,
            agentic_result,
            example,
            corpus,
            model,
            embedding_model,
        )
        if failure is not None:
            failures.append(failure)
            write_checkpoint(config, tests, outputs, evaluations, failures, "running")
            if verbose:
                print(f"[{index}/{len(tests)}] {example['id']} FAILED", flush=True)
            continue

        evaluations.append(evaluation)
        outputs.append(output)
        write_checkpoint(config, tests, outputs, evaluations, failures, "running")
        if verbose:
            print(f"[{index}/{len(tests)}] {example['id']} OK", flush=True)

    return write_checkpoint(config, tests, outputs, evaluations, failures, "completed")


def _run_example(config, agentic_result, example, corpus, model, embedding_model):
    request = _request(config, example, corpus)
    max_attempts = request.get("example_max_attempts", 1)
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        error = ValueError("example_max_attempts must be a positive integer")
        return None, None, failure_record(example, "request", error, 1)

    for attempt in range(1, max_attempts + 1):
        stage = "selection"
        try:
            component_result = select_component_skills(
                request,
                agentic_result=agentic_result,
                model=model,
                skill_root=config.skill_root,
                max_tokens=request.get("selection_max_tokens", 4096),
            )
            stage = "compilation"
            command = _compile_selected(config, component_result, model, embedding_model)
            stage = "execution"
            result = command.run(request)
            stage = "evaluation"
            evaluation = evaluate_result(example, result)
        except Exception as error:  # noqa: BLE001
            if attempt < max_attempts and stage in {"selection", "execution"}:
                continue
            return None, None, failure_record(example, stage, error, attempt)
        output = success_record(
            example, result, evaluation, component_result, attempts=attempt
        )
        return output, evaluation, None
    raise AssertionError("unreachable")


def _fixed_agentic_result(skill_root):
    specs = discover_specs(skill_root, validate_runtime=False)
    agentic = next(spec for spec in specs if spec.package_name == AGENTIC_SKILL)
    return AgenticStageResult(
        spec=agentic,
        instructions=(agentic.package_path / "SKILL.md").read_text(encoding="utf-8"),
        reason="The adaptive experiment fixes the iterative Agentic Skill.",
        advertised_skills=(AGENTIC_SKILL,),
    )


def _request(config, example, corpus):
    documents = [corpus[item] for item in example["candidate_document_ids"]]
    return {
        **config.request_defaults,
        **config.demo.request,
        "query": example["question"],
        "documents": documents,
    }


def _compile_selected(config, component_result, model, embedding_model):
    plan = RAGSelectionPlan(
        manage_skill=config.manage_skill,
        manage_guidance="Use the fixed SIM-RAG Agentic Skill.",
        manage_reason="Agentic selection is outside this experiment.",
        agentic_skill=AGENTIC_SKILL,
        agentic_reason="Compare adaptive Component plans for one iterative workflow.",
        component_bindings=component_result.bindings,
        component_reason=component_result.reason,
    )
    return compile_rag_command(
        plan,
        skill_root=config.skill_root,
        context=RuntimeComponentContext(
            executor_model=model,
            embedding_model=embedding_model,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_adaptive_experiment(load_framework_config(args.config))


if __name__ == "__main__":
    main()
