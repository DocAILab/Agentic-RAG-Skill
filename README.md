# Agentic RAG Skill

[English](README.md) · [简体中文](README.zh-CN.md)

Agentic RAG Skill is a framework for adapting retrieval-augmented generation by **selecting and revising executable Skills**. A frozen **Executor Model** selects a workflow for each question and powers its language-model components. A separately configured **Policy Model** uses Task-level feedback to revise one participating Skill in an isolated working copy. Later Tasks use the accepted revisions; model weights are never updated.

[Development guide (中文 Word)](docs/DEVELOPMENT.docx) · [Technical design and full Skill reference (中文 Word)](docs/TECHNICAL_DESIGN.docx) · [Data format (中文)](data/README.md)

## How it works

| Layer | Responsibility |
|---|---|
| **Manage** | Turn the question, corpus summary, and constraints into workflow-selection guidance. |
| **Agentic** | Define control flow and typed component slots: sequential, multi-branch, conditional, iterative, or complexity-routed execution. |
| **Component** | Implement an atomic capability: retrieval, rewriting, reranking, classification, generation, or critique. |

For each QA, the Executor reads Manage guidance, selects one Agentic Skill, then selects compatible Components for its slots. The framework checks the bindings, loads the selected Python scripts, and executes the workflow. Selection progressively exposes Skill instructions; the entire Skill library is not placed in every prompt.

After each completed Task, all selected QA contribute metrics to one optimization event. The Policy selects one participating Skill and proposes bounded file replacements. The framework checks the edit scope, Python code, and Skill contracts. With the supplied configuration, it also reruns up to three evenly spaced QA from that Task and accepts only revisions that pass the improvement gate. Failed revisions are rolled back. See the [execution and optimization design](docs/TECHNICAL_DESIGN.docx).

## Data: Group → Corpus + Tasks → QA

Each Group owns one shared Corpus and an ordered sequence of Tasks. Every QA receives the **complete Group Corpus** as its retrieval scope, including when a run is limited to a few QA; a workflow may choose to skip retrieval.

| Published Group | Tasks × QA | Corpus documents | Reference | Distractor |
|---|---:|---:|---:|---:|
| [`hotpotqa-train-001`](data/groups/hotpotqa-train-001/group.json) | 20 × 50 | 6,000 | 1,977 | 4,023 |
| [`2wiki-train-001`](data/groups/2wiki-train-001/group.json) | 20 × 50 | 6,000 | 2,311 | 3,689 |
| [`triviaqa-train-001`](data/groups/triviaqa-train-001/group.json) | 20 × 50 | 6,000 | 5,507 | 493 |

All deduplicated references are retained before distractors fill the remaining capacity. Stored records preserve annotations for audit; runtime documents expose only `id`, `title`, and `text`, in a seeded shuffled order. Answers and gold document IDs stay on the evaluation/Policy-feedback side. HotpotQA and 2Wiki use supporting-fact labels; TriviaQA uses weak answer-alias labels. Corpus sets may overlap across Groups but are not identical. FinanceBench is not included.

These sizes describe the published data, not framework limits. See the [data contract](data/README.md) and [catalog](data/groups/catalog.json) for provenance, hashes, and overlap counts.

## Install and configure

Use **Python 3.11+** and run the following commands from the repository root. An editable source checkout is the documented installation path; published data live in this checkout.

```sh
python -m pip install -e .
```

The base installation supports data validation and BM25. For the supplied model-enabled workflows and default generation evaluation, install the optional dependencies:

```sh
python -m pip install -e ".[embedding,evaluation,rerank]"
```

Copy each template to its local counterpart **only if the local file does not already exist**:

| Template → local file | Settings |
|---|---|
| [`framework/settings.example.yaml`](framework/settings.example.yaml) → `framework/settings.yaml` | Executor, optional Embedding, base Skills, request defaults, vector cache |
| [`self_optimization/settings.example.yaml`](self_optimization/settings.example.yaml) → `self_optimization/settings.yaml` | Independent Policy, revision checks, working-copy and output location |
| [`self_optimization/task_flow.example.yaml`](self_optimization/task_flow.example.yaml) → `self_optimization/task_flow.yaml` | Group, shuffle seed, Task feedback and validation sample limits |

Set the model IDs and endpoints to those offered by your service. Use `api_key_env` for a credential supplied through your shell or secret manager, with `api_key: null`; the templates reference `VVEAI_API_KEY`. Local configuration files are Git-ignored. Executor and Policy support OpenAI-compatible Chat Completions and Anthropic Messages. Embedding supports OpenAI-compatible endpoints or local Sentence Transformers; the template selects local BGE embeddings.

YAML paths resolve relative to the file containing them; CLI `--group` and `--config` paths resolve from the current directory. Local BGE, BGE reranker, GPT-2 evaluation weights, and NLTK resources may need downloading on first use. See [configuration, dependencies, and troubleshooting](docs/DEVELOPMENT.docx).

## Validate and run

Validate data without local configuration, model clients, or a run directory:

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --validate-only
```

Validate the configured pipeline and an intended selection without model calls:

```sh
python -B -m self_optimization --config self_optimization/task_flow.yaml --group data/groups/hotpotqa-train-001 --max-tasks 2 --examples-per-task 2 --validate-only
```

This checks configuration structure, paths, data, and selection. Service connectivity and model quality are assessed in separate model-enabled runs. The full Group is still validated. `group` describes the complete data; `selection` describes the chosen Task/QA IDs and counts. In data-only validation, Group fields appear at the top level.

After configuring services, run the smallest Task workflow (one QA, followed by one optimization event):

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --max-tasks 1 --examples-per-task 1
```

This invokes real models and may invoke the Executor again for revision verification. For a full Group, remove both limits:

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001
```

Switch datasets by changing only the Group path:

```sh
python -B -m self_optimization --group data/groups/2wiki-train-001
python -B -m self_optimization --group data/groups/triviaqa-train-001
```

Use `--shuffle-seed` to override Corpus order, `--config` for another task-flow configuration, and `--run-id` for a new, unique run name. Each Group invocation starts a fresh Skill copy; revisions are inherited across Tasks within that run.

All CLI entrypoints use the same Group/Task runner: `python -m self_optimization`, `python run_self_optimization.py`, `python run_task_flow.py`, `ragskill-self-optimize`, and `ragskill-task-flow`. Reinstall editable metadata after updating an older checkout to refresh console entrypoints.

## Python API

```python
from self_optimization import (
    load_task_workflow_config,
    run_self_optimization,
    validate_task_workflow,
)

config = load_task_workflow_config(
    "self_optimization/task_flow.yaml",
    group_path="data/groups/hotpotqa-train-001",
)
selection = validate_task_workflow(config, max_tasks=1, examples_per_task=1)
report = run_self_optimization(config, max_tasks=1, examples_per_task=1)
print(report["group"]["group_id"], report["completed_tasks"])
```

`run_self_optimization` aliases `run_task_workflow`; both accept `TaskWorkflowConfig`. The legacy per-sample runner is `run_sample_optimization`. For integration below the Task runner, `framework` exports `load_group`, `DataGroup` / `GroupTask` / `GroupQA`, `build_rag_request`, `select_rag_plan`, `compile_rag_command`, `run_rag`, `run_rag_from_config`, and evaluation helpers. See the [API and data contracts](docs/DEVELOPMENT.docx).

## Included Skills

The library contains **1 Manage, 5 Agentic, and 7 Component Skills**. Links lead to each package; the [complete technical reference](docs/TECHNICAL_DESIGN.docx) documents inputs, outputs, parameters, dependencies, model calls, and stopping conditions.

| Layer | Skill | Function |
|---|---|---|
| Manage | [`manage-rag-default`](framework/skills/manage/manage-rag-default) | Produce guidance for selecting an Agentic workflow. |
| Agentic | [`agentic-sequential-skill`](framework/skills/agentic/agentic-sequential-skill) | Optional rewrite → one retriever → optional rerank → generate. |
| Agentic | [`agentic-parallel-rag`](framework/skills/agentic/agentic-parallel-rag) | Combine multiple retrieval branches with reciprocal-rank fusion. |
| Agentic | [`agentic-conditional-rag`](framework/skills/agentic/agentic-conditional-rag) | Route to lexical, semantic, or fused retrieval. |
| Agentic | [`agentic-iterative-rag`](framework/skills/agentic/agentic-iterative-rag) | Accumulate evidence, generate, and critique until approved or a stop condition is reached. |
| Agentic | [`agentic-hybrid-rag`](framework/skills/agentic/agentic-hybrid-rag) | Route by complexity to no retrieval, one retrieval step, or bounded iterative retrieval. |
| Component | [`component-bm25-retriever`](framework/skills/components/component-bm25-retriever) | Field-aware BM25 over titles and bodies, with shared in-memory preprocessing. |
| Component | [`component-vector-retriever`](framework/skills/components/component-vector-retriever) | Dense cosine retrieval using the injected Embedding client and reusable indexes. |
| Component | [`component-bge-reranker`](framework/skills/components/component-bge-reranker) | Local BGE CrossEncoder reranking of retrieved candidates. |
| Component | [`component-hyde-rewriter`](framework/skills/components/component-hyde-rewriter) | Generate one hypothetical document through the Executor; requires a Vector Retriever binding. |
| Component | [`component-classifier`](framework/skills/components/component-classifier) | Executor-based retrieval-mode or question-complexity classification. |
| Component | [`component-grounded-generator`](framework/skills/components/component-grounded-generator) | Generate an answer with the Executor using supplied evidence. |
| Component | [`component-critic`](framework/skills/components/component-critic) | Ask the Executor to judge answer support and report evidence gaps. |

## Outputs and repository layout

With the supplied templates, outputs are stored beside the repository under `../Preparation/runtime/`:

| Location | Contents |
|---|---|
| `runs/<run-id>/task-flow.report.json` | Group/selection metadata, QA results, metrics, Task summaries, optimization outcomes |
| `runs/<run-id>/task-flow.events.jsonl` | Incremental workflow, Task, QA, and optimization events |
| `runs/<run-id>/skills/` and `workspace.json` | Current working Skill copy and source identity |
| `runs/<run-id>/optimizer_calls/task-XXXX/` | Policy prompts, responses, attempts, proposal, and outcome |
| `runs/<run-id>/revisions/revision-XXXX/` | Before snapshot, proposal, validation outcome, and diff when application succeeds |
| `vector-indexes/<cache-key>/` | Persistent `manifest.json` and `vectors.npy` |

BM25 preprocessing and query embeddings are cached in memory; vector document indexes can also survive process restarts. Exact reuse and invalidation rules are in the [technical design](docs/TECHNICAL_DESIGN.docx).

The repository contains runtime code (`framework/`, `self_optimization/`), executable Skills, configuration templates, final Group data, and documentation. Preparation scripts, tests, raw sources, backups, and intermediate outputs belong outside the repository. **Running the project does not import code from `Preparation`.** Offline tests with substitute models check contracts and control flow; model-enabled experiments assess real-model effectiveness separately. Maintenance and extension instructions are in the [development guide](docs/DEVELOPMENT.docx); implementation details are in the [technical design](docs/TECHNICAL_DESIGN.docx).
