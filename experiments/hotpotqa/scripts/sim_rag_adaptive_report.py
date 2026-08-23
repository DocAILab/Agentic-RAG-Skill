"""Reporting helpers for the adaptive SIM-RAG HotpotQA runner."""

import re
from datetime import UTC, datetime

from experiments.hotpotqa.scripts.run_sim_rag import (
    _answers,
    _summary,
    _write_json,
)
from framework import EvaluationExample, evaluate_example


def evaluate_result(example, result):
    return EvaluationExample(
        prediction=result["answer"],
        gold_answers=_answers(example),
        retrieved_ids=[str(item["id"]) for item in result["documents"]],
        relevant_ids=example["relevant_document_ids"],
    )


def success_record(example, result, evaluation, component_result):
    return {
        "id": example["id"],
        "type": example.get("type"),
        "question": example["question"],
        "gold_answers": _answers(example),
        "prediction": result["answer"],
        "selection": {
            "component_bindings": {
                slot: list(names) for slot, names in component_result.bindings.items()
            },
            "reason": component_result.reason,
        },
        "metrics": evaluate_example(evaluation).to_dict(),
        "retrieved_document_ids": list(evaluation.retrieved_ids),
        "relevant_document_ids": example["relevant_document_ids"],
        "trace": result.get("trace", []),
    }


def failure_record(example, stage, error):
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", str(error))
    return {
        "id": example["id"],
        "stage": stage,
        "error_type": type(error).__name__,
        "error": message,
    }


def write_checkpoint(config, tests, outputs, evaluations, failures, status):
    report = build_report(tests, outputs, evaluations, failures, status)
    _write_json(config.demo.result_path, report)
    return report


def build_report(tests, outputs, evaluations, failures, status):
    return {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "experiment": {
            "dataset": "HotpotQA distractor demo",
            "selection_mode": "adaptive-components",
            "agentic_skill": "agentic-sim-rag",
            "examples": len(tests),
            "successful_examples": len(outputs),
            "failed_examples": len(failures),
            "component_selection_counts": _selection_counts(outputs),
        },
        "summary": _summary(evaluations),
        "examples": outputs,
        "failures": failures,
    }


def _selection_counts(outputs):
    counts = {}
    for output in outputs:
        bindings = output["selection"]["component_bindings"]
        for slot, names in bindings.items():
            slot_counts = counts.setdefault(slot, {})
            for name in names or ["disabled"]:
                slot_counts[name] = slot_counts.get(name, 0) + 1
    return counts
