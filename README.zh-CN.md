<p align="center">
  <img src="docs/assets/xragos-logo.png" width="200" alt="XRAGOS 项目图标">
</p>

<h1 align="center">XRAGOS</h1>

[English](README.md) · [简体中文](README.zh-CN.md)

XRAGOS 通过**选择与修订可执行 Skill**，让检索增强生成流程适应不同问题和任务。冻结的 **Executor Model** 为每个 Task 统一规划一套工作流，让其中所有问题共用路线和组件绑定，并执行语言模型组件；独立配置的 **Policy Model** 根据 Task 级反馈，在隔离工作副本中修订一个参与执行的 Skill。后续 Task 基于已接受的修订重新规划，模型权重保持不变。

[开发文档 Word](docs/DEVELOPMENT.docx) · [技术设计与完整 Skill 说明 Word](docs/TECHNICAL_DESIGN.docx) · [数据格式](data/README.md)

## 核心工作方式

| 层级 | 职责 |
|---|---|
| **Manage** | 根据 Task 内的全部问题、Corpus 摘要和执行约束，形成统一的工作流选择指导。 |
| **Agentic** | 定义控制流和组件槽位，组织顺序、多分支、条件、迭代或复杂度路由流程。 |
| **Component** | 实现检索、改写、重排、分类、生成或评价等原子能力。 |

每个 Task 开始时，Executor 读取本次选中的全部问题，生成 Manage 指导，选择一个 Agentic Skill，再为其槽位选择兼容的 Component。这是一次完整规划中的三个顺序阶段，通常每个 Task 调用模型三次。Task 内的所有 QA 复用相同路线和绑定，同时各自执行查询、检索和答案生成。规划输入包含问题文本与语料统计；gold 答案、gold 文档 ID 和完整 Corpus 正文保留在规划输入之外。Skill 指令按层逐步提供给模型。

每个 Task 完成后，所有已选 QA 的指标共同构成一次优化反馈。Policy 选择一个参与的 Skill，提出受范围限制的文件替换；框架检查修改范围、Python 代码和 Skill 契约。使用提供的配置时，还会重跑最多三条参与该 Skill 的 QA，仅检查执行与接口契约。通过这些运行检查后保留修订；代码无法执行或接口不合法时恢复修改前的工作副本。性能指标供后续 Policy 反馈使用，不决定修订是否保留。完整过程见[技术设计](docs/TECHNICAL_DESIGN.docx)。

执行顺序为：**Task 1 统一规划 → 执行其 50 题 → 评估 → 自优化一轮 → Task 2 基于当前 Skill 重新规划 → 执行其 50 题 → 评估 → 自优化一轮**。同一轮优化中的选择、编辑和格式修复请求分别计数；执行验证重跑单独记录，不计入 50 道主评测题。报告声明 `selection_scope="task_shared"` 与 `revision_validation_mode="execution_only"`；配置项 `verify_revision_execution` 启用执行与接口检查。

Policy 的选择与编辑分别使用独立的 JSON 契约。选择响应只包含 `selected_skill` 和 `rationale`。`optimization.edit_mode: patch` 使用增量修改，`full` 使用完整文件替换；提供的示例选择 `patch`，省略此字段的旧配置保留 `full`。全文修改仅含 `path/content`；增量修改仅含 `path/base_sha256/replacements`，每个替换片段仅含 `old/new`。哈希对应可见原文统一为 LF 换行后的 UTF-8 内容，原文锚点必须唯一且不能重叠；框架先物化完整结果，再沿用原有校验和修改事务。文件权限等输入元数据不得混入输出。格式错误仅在所选模式内有界修复，不自动切换模式；原始动作、物化提案和最终差异均保留在审计记录中。

Policy 反馈包含全部已执行 QA 的指标；详细轨迹在 Task 内均匀抽样，共享方案只提供一次。字符预算不足时减少完整轨迹条目，并记录保留与省略数量。该限制只作用于反馈体积，主 QA 仍全部执行，可编辑 Skill 的源码保持完整。

## 数据组织 Group → Corpus + Tasks → QA

每个 Group 包含一份共享 Corpus 和一组有序 Task。每个 QA 的检索范围都是**完整 Group Corpus**，即使本次只运行少量 QA；工作流可以根据路线选择是否检索。

| 数据集 | Group 数 | 覆盖的独立训练 QA | QA 总出现次数 | 每组 Reference 范围 |
|---|---:|---:|---:|---:|
| HotpotQA | 91 | 90,447 | 91,000 | 1,968–1,991 |
| 2WikiMultihopQA | 168 | 167,454 | 168,000 | 1,917–3,546 |
| TriviaQA RC | 102 | 76,220 | 102,000 | 973–6,000 |

当前共 **361 个 Group，覆盖 334,121 条独立有效训练 QA**；计入跨 Group 复用后，共 361,000 条 QA。每组统一为 **20 个 Task × 50 个 QA、6,000 篇 Corpus**。各 Group ID 和准确的 reference/distractor 数量见 [catalog](data/groups/catalog.json)。

构建时先保留全部去重 reference，再用 distractor 补足容量。离线文件保留标注以便审计；运行时文档只暴露 `id`、`title` 和 `text`，并按种子打乱顺序。答案和 gold 文档 ID 保留在评估与 Policy 反馈侧。HotpotQA 和 2Wiki 使用 supporting-fact 标签，TriviaQA 使用答案别名匹配得到的弱标签。不同 Group 的 Corpus 可以部分重叠，但完整集合不同；当前发布数据不包含 FinanceBench。

采样优先覆盖尚未使用的 QA，保留其全部 reference；必要时复用符合容量约束的 QA，把每组补足到 1,000 题。每个新 Group 的 Corpus 内容集合不同，并记录选样策略、QA 与语料重叠。原有三个 `*-train-001` 文件包保持不变。表中规模是当前发布数据的参数。数据来源、哈希与重叠统计见[数据契约](data/README.md)和 [catalog](data/groups/catalog.json)。

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

根据模型服务填写模型 ID 和 API 地址。通过 shell 或密钥管理工具提供凭据，在配置中使用 `api_key_env`，同时保持 `api_key: null`；模板引用的环境变量名为 `DASHSCOPE_API_KEY`。本地配置文件已被 Git 忽略。Executor 和 Policy 支持 OpenAI-compatible Chat Completions 与 Anthropic Messages；Embedding 支持 OpenAI-compatible 接口或本地 Sentence Transformers，模板使用本地 BGE 编码器。

默认 Policy 目前使用百炼 API 的 `qwen3.5-27b`；本地 `qwen3.5-9b` 配置继续保留，需单独部署服务后再切换。

Policy 通过 `optimizer.mode` 显式选择 `local`（自托管模型）或 `api`（云 API），两套服务参数分别放在 `optimizer.profiles.local` 和 `optimizer.profiles.api`。单次运行可用 `--policy-mode local` 或 `--policy-mode api` 覆盖保存的选择。默认本地配置指向 `http://127.0.0.1:18081/v1`，模型别名为 `qwen3.5-9b`，需先启动对应模型服务；云 API 则需确认模型已对当前密钥开放。两种模式不会在失败时自动切换端点或模型。`options.extra_body.stream: true` 可启用 OpenAI 流式响应，框架汇总完整文本后再验证提案。

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

同一 Task 的 QA 可以共用规划和 Skill 版本并行执行。`--qa-mode adaptive --qa-max-workers 10` 根据持续吞吐量和真实重试压力调整有上限的并发数，单题延迟不触发降并发；`--qa-mode fixed --qa-max-workers 4` 使用固定上限；`--qa-mode serial` 保持串行。全部 QA 完成后才进行优化和下一 Task。配置、统计、共享资源与自定义客户端要求见[Task 内并行说明](docs/PARALLEL_EXECUTION.md)。

自适应模式可复用紧邻的已提交 Task 在相同服务、路线和资源下验证过的健康并发水平，降低一级后作为新起点，并在新 Task 重新测量吞吐。`workflow.qa_execution.reuse_healthy_start: false` 可关闭复用。发生重试压力、统计覆盖不足或执行身份变化时，按配置冷启动。

续跑中断实验时，保持原有配置、Group、选取数量、源码版本和 run ID，并增加 `--resume`：

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --run-id experiment-001 --resume
```

已提交的 Task 保留结果；未完成的 Task 恢复其起始 Skill 快照，重新规划并重跑该 Task，原有部分回答保留为诊断记录。`--freeze-skills` 使用同样的 Task 级统一规划，同时不构造或调用 Policy。`--task-budget-seconds` 在下一个完整 Task 边界暂停并返回退出码 75，之后可通过 `--resume` 继续。

使用 `python -m self_optimization` 或安装后的 `ragskill-task-flow` 命令，二者运行相同的 Group/Task 工作流。更新源码后，重新执行可编辑安装以刷新 console entrypoint。

## 测试工作流可视化

启动本地面板，读取已有实验及正在写入的运行产物：

```bash
python -B -m visualization --runtime-root ../Preparation/runtime
```

浏览器打开 `http://127.0.0.1:8765`，选择实验和 Task，即可沿着“输入 → Manage → Agentic → Component → QA 执行 → 评估 → Policy 选择与修改 → 验证 → 下一 Task”检查真实记录。点击环节查看输入、完整 prompt、实际模型及回答、代码逻辑、修改差异和 Skill 哈希继承；选择任意 QA 查看其结果和执行轨迹。逻辑优化轮数、模型 generate 请求与 HTTP 尝试分别显示，验证重跑单独计数。

面板只读本地产物，不调用模型。默认使用手动刷新；可用 `--port` 更换端口。安装源码后也可使用 `ragskill-visualize`。旧运行缺失的 prompt 或传输统计会明确标为未记录，不从当前源码补造历史。完整使用说明见[可视化说明](docs/VISUALIZATION.md)。

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

`run_task_workflow` 接收 `TaskWorkflowConfig`，运行一个 Group 内的有序 Task。需要集成更底层能力时，可从 `framework` 导入 `load_group`、`DataGroup` / `GroupTask` / `GroupQA`、`build_rag_request`、`select_rag_plan`、`compile_rag_command`、`run_rag`、`run_rag_from_config` 和评估函数。接口与数据契约见[开发文档](docs/DEVELOPMENT.docx)。

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
| `runs/<run-id>/model-captures/` | 脱敏模型输入输出、逐组件函数记录、执行源码快照及采集来源 |
| `runs/<run-id>/task_routes/<task>/<attempt>/` | 含 Task 全部问题的一次规划输入、共享方案及哈希 |
| `runs/<run-id>/checkpoints/` | Task 持久化提交、起始快照、中断尝试与续跑身份 |
| `runs/<run-id>/skills/` 与 `workspace.json` | 当前 Skill 工作副本与来源身份 |
| `runs/<run-id>/optimizer_calls/task-XXXX/` | Policy 提示、响应、尝试、提案与处理结果 |
| `runs/<run-id>/revisions/revision-XXXX/` | 修改前快照、提案、校验结果，以及应用成功后生成的差异 |
| `vector-indexes/<cache-key>/` | 持久化的 `manifest.json` 与 `vectors.npy` |

BM25 预处理和查询向量保存在内存缓存中，文档向量索引还可以跨进程复用。具体复用与失效规则见[技术文档](docs/TECHNICAL_DESIGN.docx)。

仓库包含运行代码（`framework/`、`self_optimization/`）、可执行 Skill、配置模板、最终 Group 数据和正式文档。准备脚本、测试、原始数据、备份与中间产物放在仓库外。**正式运行不导入 `Preparation` 中的代码。** 离线替身测试验证契约与控制流，真实模型实验单独评估模型效果。维护和扩展方法见[开发文档](docs/DEVELOPMENT.docx)，实现细节见[技术文档](docs/TECHNICAL_DESIGN.docx)。
