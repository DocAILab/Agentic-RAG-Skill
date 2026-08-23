"""Run the HotpotQA demo with one fixed SIM-RAG component plan."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from experiments.hotpotqa.scripts.metrics import (
    evaluate_hotpotqa,
    summarize_hotpotqa,
)
from framework import (
    RAGSelectionPlan,
    RuntimeComponentContext,
    compile_rag_command,
    create_clients_from_config,
    load_framework_config,
)

FIXED_BINDINGS = {
    "rewriter": (),
    "retriever": ("component-bm25-retriever",),
    "reranker": (),
    "generator": ("component-grounded-generator",),
    "critic": ("component-critic",),
}


def run_experiment(config, *, model=None, embedding_model=None, verbose=True):
    """Execute the fixed plan and write a schema-v2 report."""
    demo = config.demo
    if demo is None:
        raise ValueError("Experiment config requires a demo section")
    if model is None:
        model, configured_embedding = create_clients_from_config(config)
        if embedding_model is None:
            embedding_model = configured_embedding
    command = _compile_command(config, model, embedding_model)
    tests = _load_jsonl(demo.test_path)[: demo.max_examples]
    corpus = {item["id"]: item for item in _load_jsonl(demo.corpus_path)}
    outputs = []
    evaluations = []
    failures = []
    for index, example in enumerate(tests, start=1):
        documents = [corpus[item] for item in example["candidate_document_ids"]]
        request = {
            **config.request_defaults,
            **demo.request,
            "query": example["question"],
            "documents": documents,
        }
        try:
            result = command.run(request)
        except Exception as error:  # noqa: BLE001
            failures.append(_failure_record(example, error))
            _checkpoint(config, tests, outputs, evaluations, failures, "running")
            if verbose:
                print(f"[{index}/{len(tests)}] {example['id']} FAILED", flush=True)
            continue
        retrieved_ids = [str(item["id"]) for item in result["documents"]]
        metrics = evaluate_hotpotqa(
            result["answer"],
            _answers(example),
            retrieved_ids,
            example["relevant_document_ids"],
        )
        evaluations.append(metrics)
        outputs.append(_success_record(example, result, retrieved_ids, metrics))
        _checkpoint(config, tests, outputs, evaluations, failures, "running")
        if verbose:
            print(f"[{index}/{len(tests)}] {example['id']} OK", flush=True)
    report = _report(config, tests, outputs, evaluations, failures, "completed")
    _write_json(demo.result_path, report)
    return report


def _compile_command(config, model, embedding_model):
    plan = RAGSelectionPlan(
        manage_skill=config.manage_skill,
        manage_guidance="Fixed HotpotQA experiment plan.",
        manage_reason="Selection is outside the experimental variable.",
        agentic_skill="agentic-sim-rag",
        agentic_reason="Controlled iterative workflow.",
        component_bindings=FIXED_BINDINGS,
        component_reason="Fixed BM25, grounded Generator, and Critic.",
    )
    return compile_rag_command(
        plan,
        skill_root=config.skill_root,
        context=RuntimeComponentContext(
            executor_model=model,
            embedding_model=embedding_model,
        ),
    )


def _success_record(example, result, retrieved_ids, metrics):
    return {
        "id": example["id"],
        "type": example.get("type"),
        "question": example["question"],
        "gold_answers": _answers(example),
        "prediction": result["answer"],
        "metrics": metrics,
        "retrieved_document_ids": retrieved_ids,
        "relevant_document_ids": example["relevant_document_ids"],
        "trace": result.get("trace", []),
    }


def _failure_record(example, error):
    message = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", str(error))
    return {
        "id": example["id"],
        "error_type": type(error).__name__,
        "error": message,
    }


def _checkpoint(config, tests, outputs, evaluations, failures, status):
    report = _report(config, tests, outputs, evaluations, failures, status)
    _write_json(config.demo.result_path, report)


def _report(config, tests, outputs, evaluations, failures, status):
    request = config.demo.request
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "experiment": {
            "dataset": "HotpotQA distractor demo",
            "examples": len(tests),
            "successful_examples": len(outputs),
            "failed_examples": len(failures),
            "bindings": {key: list(value) for key, value in FIXED_BINDINGS.items()},
            "top_k": request.get("top_k", 3),
            "max_iterations": request.get("max_iterations", 3),
            "max_tokens": request.get("max_tokens"),
            "critic_max_tokens": request.get("critic_max_tokens", 4096),
        },
        "summary": _summary(evaluations),
        "examples": outputs,
        "failures": failures,
    }


def _summary(evaluations):
    return summarize_hotpotqa(evaluations)


def _answers(example):
    if "answers" in example:
        return list(example["answers"])
    return [example["answer"]]


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(load_framework_config(args.config))


if __name__ == "__main__":
    main()
