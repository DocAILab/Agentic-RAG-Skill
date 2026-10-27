# Agentic RAG Skill

[English](README.md) · [简体中文](README.zh-CN.md)

Agentic RAG Skill 通过**选择与修订可执行 Skill**，让检索增强生成流程适应不同问题和任务。冻结的 **Executor Model** 为每个问题选择工作流，并执行其中的语言模型组件；独立配置的 **Policy Model** 根据 Task 级反馈，在隔离工作副本中修订一个参与执行的 Skill。后续 Task 使用已接受的修订，模型权重保持不变。

[开发文档 Word](docs/DEVELOPMENT.docx) · [技术设计与完整 Skill 说明 Word](docs/TECHNICAL_DESIGN.docx) · [数据格式](data/README.md)

## 核心工作方式

| 层级 | 职责 |
|---|---|
| **Manage** | 根据问题、Corpus 摘要和约束，形成工作流选择指导。 |
| **Agentic** | 定义控制流和组件槽位，组织顺序、多分支、条件、迭代或复杂度路由流程。 |
| **Component** | 实现检索、改写、重排、分类、生成或评价等原子能力。 |

每个 QA 都由 Executor 读取 Manage 指导，选择一个 Agentic Skill，再为其槽位选择兼容的 Component。框架校验绑定关系，加载选中的 Python 脚本并执行工作流。Skill 指令按层逐步提供给模型。

每个 Task 完成后，所有已选 QA 的指标共同构成一次优化反馈。Policy 选择一个参与的 Skill，提出受范围限制的文件替换；框架检查修改范围、Python 代码和 Skill 契约。使用提供的配置时，还会从该 Task 均匀选取最多三题重新运行，只接受通过指标提升判据的修订，未通过的修订会回滚。完整过程见[技术设计](docs/TECHNICAL_DESIGN.docx)。

## 数据组织 Group → Corpus + Tasks → QA

每个 Group 包含一份共享 Corpus 和一组有序 Task。每个 QA 的检索范围都是**完整 Group Corpus**，即使本次只运行少量 QA；工作流可以根据路线选择是否检索。

| 已发布 Group | Task × QA | Corpus 文档数 | Reference | Distractor |
|---|---:|---:|---:|---:|
| [`hotpotqa-train-001`](data/groups/hotpotqa-train-001/group.json) | 20 × 50 | 6,000 | 1,977 | 4,023 |
| [`2wiki-train-001`](data/groups/2wiki-train-001/group.json) | 20 × 50 | 6,000 | 2,311 | 3,689 |
| [`triviaqa-train-001`](data/groups/triviaqa-train-001/group.json) | 20 × 50 | 6,000 | 5,507 | 493 |

构建时先保留全部去重 reference，再用 distractor 补足容量。离线文件保留标注以便审计；运行时文档只暴露 `id`、`title` 和 `text`，并按种子打乱顺序。答案和 gold 文档 ID 保留在评估与 Policy 反馈侧。HotpotQA 和 2Wiki 使用 supporting-fact 标签，TriviaQA 使用答案别名匹配得到的弱标签。不同 Group 的 Corpus 可以部分重叠，但完整集合不同；当前发布数据不包含 FinanceBench。

表中规模是当前发布数据的参数。数据来源、哈希与重叠统计见[数据契约](data/README.md)和 [catalog](data/groups/catalog.json)。

## 安装与配置

使用 **Python 3.11+**，以下命令均在仓库根目录执行。推荐以可编辑模式安装源码，正式数据随当前源码仓库提供。

```sh
python -m pip install -e .
```

基础安装支持数据校验和 BM25。运行提供的模型工作流及默认生成评估时，安装对应可选依赖：

```sh
python -m pip install -e ".[embedding,evaluation,rerank]"
```

将模板复制为对应本地文件，**仅在目标文件尚不存在时复制**：

| 模板 → 本地文件 | 配置内容 |
|---|---|
| [`framework/settings.example.yaml`](framework/settings.example.yaml) → `framework/settings.yaml` | Executor、可选 Embedding、基础 Skill、请求默认值、向量缓存 |
| [`self_optimization/settings.example.yaml`](self_optimization/settings.example.yaml) → `self_optimization/settings.yaml` | 独立 Policy、修订校验、工作副本和输出位置 |
| [`self_optimization/task_flow.example.yaml`](self_optimization/task_flow.example.yaml) → `self_optimization/task_flow.yaml` | Group、打乱种子、Task 反馈与验证样本上限 |

根据模型服务填写模型 ID 和 API 地址。通过 shell 或密钥管理工具提供凭据，在配置中使用 `api_key_env`，同时保持 `api_key: null`；模板引用的环境变量名为 `VVEAI_API_KEY`。本地配置文件已被 Git 忽略。Executor 和 Policy 支持 OpenAI-compatible Chat Completions 与 Anthropic Messages；Embedding 支持 OpenAI-compatible 接口或本地 Sentence Transformers，模板使用本地 BGE 编码器。

YAML 中的路径相对于声明它的配置文件解析；CLI 的 `--group` 和 `--config` 相对于当前工作目录解析。本地 BGE、BGE reranker、GPT-2 评估权重和 NLTK 资源在首次使用时可能需要下载。配置、依赖与排障方法见[开发文档](docs/DEVELOPMENT.docx)。

## 校验与运行

只校验数据，无需本地模型配置，也不会创建模型客户端或运行目录：

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --validate-only
```

校验配置链及计划运行的范围，同样不调用模型：

```sh
python -B -m self_optimization --config self_optimization/task_flow.yaml --group data/groups/hotpotqa-train-001 --max-tasks 2 --examples-per-task 2 --validate-only
```

该命令检查配置结构、路径、数据和选取范围；服务连通性与模型效果通过单独的真实模型运行评估。完整 Group 的数据仍会被校验。输出中的 `group` 描述完整数据，`selection` 描述本次选中的 Task/QA ID 和数量；纯数据校验时，Group 字段直接位于顶层。

配置好模型服务后，运行最小 Task 流程，即一个 QA 加一次 Task 优化：

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --max-tasks 1 --examples-per-task 1
```

这会调用真实模型，修订验证还可能再次调用 Executor。运行完整 Group 时去掉两个数量限制：

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001
```

切换数据集只需修改 Group 路径：

```sh
python -B -m self_optimization --group data/groups/2wiki-train-001
python -B -m self_optimization --group data/groups/triviaqa-train-001
```

用 `--shuffle-seed` 覆盖 Corpus 打乱种子，用 `--config` 指定其他任务配置，用 `--run-id` 指定新的唯一运行名称。每次 Group 调用都创建新的 Skill 工作副本，同一次运行内的后续 Task 继承已接受的修订。

所有 CLI 入口使用同一个 Group/Task runner：`python -m self_optimization`、`python run_self_optimization.py`、`python run_task_flow.py`、`ragskill-self-optimize` 和 `ragskill-task-flow`。更新较早安装的项目后，重新执行可编辑安装以刷新 console entrypoint。

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

`run_self_optimization` 是 `run_task_workflow` 的别名，二者均接收 `TaskWorkflowConfig`。兼容的逐样本入口为 `run_sample_optimization`。需要集成更底层能力时，可从 `framework` 导入 `load_group`、`DataGroup` / `GroupTask` / `GroupQA`、`build_rag_request`、`select_rag_plan`、`compile_rag_command`、`run_rag`、`run_rag_from_config` 和评估函数。接口与数据契约见[开发文档](docs/DEVELOPMENT.docx)。

## Skill 总览

当前库包含 **1 个 Manage、5 个 Agentic、7 个 Component Skill**。下表链接指向各包；[完整技术文档](docs/TECHNICAL_DESIGN.docx)逐个说明输入、输出、参数、依赖、模型调用和停止条件。

| 层级 | Skill | 作用 |
|---|---|---|
| Manage | [`manage-rag-default`](framework/skills/manage/manage-rag-default) | 为 Agentic 工作流选择生成指导。 |
| Agentic | [`agentic-sequential-skill`](framework/skills/agentic/agentic-sequential-skill) | 可选改写 → 单检索器 → 可选重排 → 生成。 |
| Agentic | [`agentic-parallel-rag`](framework/skills/agentic/agentic-parallel-rag) | 使用 reciprocal-rank fusion 合并多条检索分支。 |
| Agentic | [`agentic-conditional-rag`](framework/skills/agentic/agentic-conditional-rag) | 路由到词法、语义或融合检索。 |
| Agentic | [`agentic-iterative-rag`](framework/skills/agentic/agentic-iterative-rag) | 累积证据、生成并评价，直到获批或满足停止条件。 |
| Agentic | [`agentic-hybrid-rag`](framework/skills/agentic/agentic-hybrid-rag) | 根据复杂度选择不检索、单步检索或有界迭代检索。 |
| Component | [`component-bm25-retriever`](framework/skills/components/component-bm25-retriever) | 对标题与正文进行分字段 BM25 检索，共享内存预处理。 |
| Component | [`component-vector-retriever`](framework/skills/components/component-vector-retriever) | 使用注入的 Embedding 客户端和可复用索引进行稠密余弦检索。 |
| Component | [`component-bge-reranker`](framework/skills/components/component-bge-reranker) | 使用本地 BGE CrossEncoder 对候选文档重排。 |
| Component | [`component-hyde-rewriter`](framework/skills/components/component-hyde-rewriter) | 通过 Executor 生成一篇假设文档，需要绑定 Vector Retriever。 |
| Component | [`component-classifier`](framework/skills/components/component-classifier) | 通过 Executor 判断检索模式或问题复杂度。 |
| Component | [`component-grounded-generator`](framework/skills/components/component-grounded-generator) | 通过 Executor 根据提供的证据生成答案。 |
| Component | [`component-critic`](framework/skills/components/component-critic) | 通过 Executor 评价答案的证据支持并指出证据缺口。 |

## 输出与仓库结构

使用提供的模板时，产物保存在仓库同级的 `../Preparation/runtime/`：

| 位置 | 内容 |
|---|---|
| `runs/<run-id>/task-flow.report.json` | Group/选取范围、QA 结果、指标、Task 汇总、优化结果 |
| `runs/<run-id>/task-flow.events.jsonl` | 按阶段追加的 workflow、Task、QA 和优化事件 |
| `runs/<run-id>/skills/` 与 `workspace.json` | 当前 Skill 工作副本与来源身份 |
| `runs/<run-id>/optimizer_calls/task-XXXX/` | Policy 提示、响应、尝试、提案与处理结果 |
| `runs/<run-id>/revisions/revision-XXXX/` | 修改前快照、提案、校验结果，以及应用成功后生成的差异 |
| `vector-indexes/<cache-key>/` | 持久化的 `manifest.json` 与 `vectors.npy` |

BM25 预处理和查询向量保存在内存缓存中，文档向量索引还可以跨进程复用。具体复用与失效规则见[技术文档](docs/TECHNICAL_DESIGN.docx)。

仓库包含运行代码（`framework/`、`self_optimization/`）、可执行 Skill、配置模板、最终 Group 数据和正式文档。准备脚本、测试、原始数据、备份与中间产物放在仓库外。**正式运行不导入 `Preparation` 中的代码。** 离线替身测试验证契约与控制流，真实模型实验单独评估模型效果。维护和扩展方法见[开发文档](docs/DEVELOPMENT.docx)，实现细节见[技术文档](docs/TECHNICAL_DESIGN.docx)。
