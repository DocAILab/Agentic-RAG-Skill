<p align="center">
  <img src="docs/assets/xragos-logo.png" width="200" alt="XRAGOS logo">
</p>

<h1 align="center">XRAGOS</h1>

[English](README.md) · [简体中文](README.zh-CN.md)

XRAGOS is a framework for adapting retrieval-augmented generation by **selecting and revising executable Skills**. A frozen **Executor Model** selects one shared workflow for all questions in a Task and powers its language-model components. A separately configured **Policy Model** uses Task-level feedback to revise one participating Skill in an isolated working copy. Later Tasks plan again using the accepted revisions; model weights are never updated.

[Development guide (中文 Word)](docs/DEVELOPMENT.docx) · [Technical design and full Skill reference (中文 Word)](docs/TECHNICAL_DESIGN.docx) · [Data format (中文)](data/README.md)

## How it works

| Layer | Responsibility |
|---|---|
| **Manage** | Turn all Task questions, the corpus summary, and execution constraints into shared workflow-selection guidance. |
| **Agentic** | Define control flow and typed component slots: sequential, multi-branch, conditional, iterative, or complexity-routed execution. |
| **Component** | Implement an atomic capability: retrieval, rewriting, reranking, classification, generation, or critique. |

At the start of each Task, the Executor reads all selected questions, produces Manage guidance, selects one Agentic Skill, and binds compatible Components. These are three sequential model stages, normally three requests per Task. Every QA in that Task reuses the same route and bindings while retaining its own query, retrieval, and answer generation. Routing receives question text and corpus statistics; gold answers, gold document IDs, and full corpus text stay outside routing prompts. Selection progressively exposes Skill instructions.

After each completed Task, all selected QA contribute metrics to one optimization event. The Policy selects one participating Skill and proposes bounded file replacements. The framework checks the edit scope, Python code, and Skill contracts. With the supplied configuration, it also replays up to three participating QA to check execution and interface contracts. Revisions are retained when these operational checks pass; execution or contract failures restore the previous working copy. Performance metrics inform later Policy feedback and do not decide whether a revision is retained. See the [execution and optimization design](docs/TECHNICAL_DESIGN.docx).

The execution order is **Task 1 shared plan → its 50 QA → evaluation → one logical optimization round → Task 2 shared plan using the resulting Skills → its 50 QA → evaluation → one logical optimization round**. Selection, editing, and format-repair requests are counted separately within that optimization round. Execution-validation replays are recorded separately from the 50 primary QA. Reports declare `selection_scope="task_shared"` and `revision_validation_mode="execution_only"`; `verify_revision_execution` enables execution and interface checks.

Policy selection and editing have separate JSON contracts. Selection returns only `selected_skill` and `rationale`. Set `optimization.edit_mode` to `patch` for exact incremental edits, or `full` for complete file replacements. The supplied example selects `patch`; configurations omitting this field retain `full`. Full edits contain exactly `path` and `content`. Patch edits contain exactly `path`, `base_sha256`, and `replacements`, each with exact `old`/`new` text. The hash identifies the visible LF-normalized original; anchors must match once and cannot overlap. The framework materializes and validates the complete result before applying it. Input metadata is never part of either output contract. Format errors receive bounded repair attempts in the selected mode; modes never fall back automatically. Original actions, materialized proposals and final diffs remain in the audit record.

Policy feedback includes every executed QA's metrics. Detailed traces are sampled across the Task and share one plan; the character budget may reduce the number of complete trace examples, with included and omitted counts recorded. This changes feedback size only. Primary QA execution and the complete editable Skill source are preserved.

## Data: Group → Corpus + Tasks → QA

Each Group owns one shared Corpus and an ordered sequence of Tasks. Every QA receives the **complete Group Corpus** as its retrieval scope, including when a run is limited to a few QA; a workflow may choose to skip retrieval.

| Dataset | Groups | Unique train QA covered | QA occurrences | References per Group |
|---|---:|---:|---:|---:|
| HotpotQA | 91 | 90,447 | 91,000 | 1,968–1,991 |
| 2WikiMultihopQA | 168 | 167,454 | 168,000 | 1,917–3,546 |
| TriviaQA RC | 102 | 76,220 | 102,000 | 973–6,000 |

The release contains **361 Groups covering 334,121 unique valid training QA**, with 361,000 QA occurrences including cross-Group reuse. Every Group has **20 Tasks × 50 QA and 6,000 Corpus documents**. Group IDs and exact per-Group reference/distractor counts are listed in the [catalog](data/groups/catalog.json).

All deduplicated references are retained before distractors fill the remaining capacity. Stored records preserve annotations for audit; runtime documents expose only `id`, `title`, and `text`, in a seeded shuffled order. Answers and gold document IDs stay on the evaluation/Policy-feedback side. HotpotQA and 2Wiki use supporting-fact labels; TriviaQA uses weak answer-alias labels. Corpus sets may overlap across Groups but are not identical. FinanceBench is not included.

Sampling prioritizes uncovered QA, retains all of their references, and reuses capacity-compatible QA when needed to reach 1,000 questions without exceeding 6,000 documents. Each new Group has a distinct Corpus and records its selection policy; cross-Group QA and Corpus overlap are measured. The original three `*-train-001` packages are unchanged. These sizes describe the published data, not framework limits. See the [data contract](data/README.md) and [catalog](data/groups/catalog.json) for provenance, hashes, and overlap counts.

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

Set the model IDs and endpoints to those offered by your service. Use `api_key_env` for a credential supplied through your shell or secret manager, with `api_key: null`; the templates reference `DASHSCOPE_API_KEY`. Local configuration files are Git-ignored. Executor and Policy support OpenAI-compatible Chat Completions and Anthropic Messages. Embedding supports OpenAI-compatible endpoints or local Sentence Transformers; the template selects local BGE embeddings.

The default Policy is now API mode with `qwen3.5-27b` on Alibaba Cloud Model Studio; the local `qwen3.5-9b` profile remains available and requires a separately deployed service.

Policy supports explicit `local` and `api` modes in `optimizer.mode`, with separate service settings under `optimizer.profiles.local` and `optimizer.profiles.api`. Override the saved choice for one run with `--policy-mode local` or `--policy-mode api`. The local template targets a self-hosted `qwen3.5-9b` at `http://127.0.0.1:18081/v1`; deploy that service first and match its model alias. API models must be available to your credential. Modes never fall back to another endpoint or model. `options.extra_body.stream: true` enables OpenAI SSE responses, which are collected into complete text before proposal validation.

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

QA within one Task can run concurrently while sharing its plan and Skill version. Use `--qa-mode adaptive --qa-max-workers 10` for bounded adjustment based on sustained throughput and actual retry pressure, `--qa-mode fixed --qa-max-workers 4` for a fixed limit, or `--qa-mode serial`. Per-question latency does not trigger a concurrency decrease. Task optimization and the next Task wait for all QA to finish. See [parallel execution](docs/PARALLEL_EXECUTION.md) for configuration, measurements, resource sharing, and custom-client requirements.

Adaptive execution can reuse a conservative starting limit from the immediately preceding committed Task with the same service, route and resources. Only fully observed healthy windows qualify; the tested limit is reduced by one and throughput is measured again in the new Task. Set `workflow.qa_execution.reuse_healthy_start: false` to disable reuse. Retry pressure, incomplete statistics and changed execution identity cause a cold start.

To continue an interrupted run, keep the original configuration, Group, selection limits, source version, and run ID, and add `--resume`:

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --run-id experiment-001 --resume
```

Completed Tasks remain committed. An incomplete Task restores its starting Skill snapshot, plans again, and reruns that Task; its earlier partial answers remain diagnostic evidence. `--freeze-skills` runs the same Task-shared routing without constructing or calling the Policy model. `--task-budget-seconds` pauses at the next completed Task boundary with exit code 75, allowing continuation with `--resume`.

Use `python -m self_optimization` or the installed `ragskill-task-flow` command; both run the same Group/Task workflow. Reinstall the editable package after updating a checkout to refresh console entrypoints.

## Visualize test workflows

Start the local viewer for existing experiments and runs in progress:

```bash
python -B -m visualization --runtime-root ../Preparation/runtime
```

Open `http://127.0.0.1:8765`, select a run and Task, then follow input → Manage → Agentic → Component → QA execution → evaluation → Policy selection/editing → validation → the next Task. Click a stage to inspect inputs, full prompts, actual models and responses, execution code, revision diffs, and inherited Skill hashes. Individual QA results and traces are selectable. Logical optimization rounds, model generate requests, HTTP attempts, and validation replays remain separate counts.

The viewer reads local artifacts without calling models. Refresh is manual by default; change the port with `--port`. Installed checkouts also expose `ragskill-visualize`. Missing prompts or transport statistics in older runs are labeled unrecorded; current source code is never presented as historical execution evidence. See the [viewer guide](docs/VISUALIZATION.md).

## Python API

```python
from self_optimization import (
    load_task_workflow_config,
    run_task_workflow,
    validate_task_workflow,
)

config = load_task_workflow_config(
    "self_optimization/task_flow.yaml",
    group_path="data/groups/hotpotqa-train-001",
)
selection = validate_task_workflow(config, max_tasks=1, examples_per_task=1)
report = run_task_workflow(config, max_tasks=1, examples_per_task=1)
print(report["group"]["group_id"], report["completed_tasks"])
```

`run_task_workflow` accepts `TaskWorkflowConfig` for one Group and its ordered Tasks. For integration below the Task runner, `framework` exports `load_group`, `DataGroup` / `GroupTask` / `GroupQA`, `build_rag_request`, `select_rag_plan`, `compile_rag_command`, `run_rag`, `run_rag_from_config`, and evaluation helpers. See the [API and data contracts](docs/DEVELOPMENT.docx).

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
| `runs/<run-id>/model-captures/` | Redacted model I/O, per-component inputs/outputs, execution source snapshots and capture provenance |
| `runs/<run-id>/task_routes/<task>/<attempt>/` | One routing request containing all Task questions, the shared plan, and hashes |
| `runs/<run-id>/checkpoints/` | Durable Task commits, starting snapshots, interrupted attempts, and resume identity |
| `runs/<run-id>/skills/` and `workspace.json` | Current working Skill copy and source identity |
| `runs/<run-id>/optimizer_calls/task-XXXX/` | Policy prompts, responses, attempts, proposal, and outcome |
| `runs/<run-id>/revisions/revision-XXXX/` | Before snapshot, proposal, validation outcome, and diff when application succeeds |
| `vector-indexes/<cache-key>/` | Persistent `manifest.json` and `vectors.npy` |

BM25 preprocessing and query embeddings are cached in memory; vector document indexes can also survive process restarts. Exact reuse and invalidation rules are in the [technical design](docs/TECHNICAL_DESIGN.docx).

The repository contains runtime code (`framework/`, `self_optimization/`), executable Skills, configuration templates, final Group data, and documentation. Preparation scripts, tests, raw sources, backups, and intermediate outputs belong outside the repository. **Running the project does not import code from `Preparation`.** Offline tests with substitute models check contracts and control flow; model-enabled experiments assess real-model effectiveness separately. Maintenance and extension instructions are in the [development guide](docs/DEVELOPMENT.docx); implementation details are in the [technical design](docs/TECHNICAL_DESIGN.docx).
