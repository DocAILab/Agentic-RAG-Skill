# Agentic RAG Skill

## Skill 自优化

`self_optimization/` 实现独立于基础 Skill 仓库的逐样本自优化循环：每条查询完成检索和生成后，先使用现有 XRAG 对齐评估器计算指标，再把评估结果、Skill 选择、检索轨迹、组件耗时和有界文档片段交给独立 Policy Model。优化采用两阶段按需加载：第一阶段只暴露本轮候选 Skill 的名称、层级和能力摘要，让 Policy Model 选择一个 Skill；第二阶段只加载被选 Skill 的完整文件，由同一 Policy Model 提交文件替换。编辑阶段不会看到其他候选 Skill 的源码。

每次运行都会将 `framework/skills/` 完整复制到新的 `self_optimization/runs/<run-id>/skills/`。优化只写入该副本；`ragskill.yaml`、包名和运行时契约受保护。每轮修改前保存快照，修改后重新验证整个 Skill 仓库和 Python 入口，非法修改自动回滚。Python 修改还需通过 Ruff 语法、未定义名称、未使用导入检查；Agentic workflow 只能调用清单声明的抽象组件槽位；Component 的字面量返回值必须保留其 `output_type` 声明的必需字段，例如 Retriever 必须返回 `documents`，Generator 必须返回 `answer`。修订内容、统一 diff、逐阶段日志和最终报告均保存在对应运行目录。每次 Policy Model 调用还会在 `optimizer_calls/` 下保存完整 `system_prompt.txt`、`prompt.txt`、`raw_response.txt`、解析结果和最终 `result.json`，并以 `selection` 或 `editing` 标注调用阶段。提案若使用错误文件路径、没有实质修改、违反能力契约或返回非法 JSON，框架会把累积错误反馈给模型进行有限次数的自动纠正；每次尝试分别保存在 `optimizer_calls/*/attempts/attempt-XXXX/`。

优化结果分为三种状态：`accepted` 表示副本中产生了通过验证的 revision，`skipped` 表示 Policy Model 已选择目标但在有限次编辑中没有提出实质变化，`rejected` 仅表示非法提案、契约违规或模型/API 错误。`skipped` 不会修改 Skill，也不会阻止后续 task 继续执行。

```powershell
Copy-Item self_optimization/settings.example.yaml self_optimization/settings.yaml
# 在 settings.yaml 中填写本地 Qwen OpenAI-compatible API 地址和真实模型 ID
python -B run_self_optimization.py --limit 10
```

模板默认使用 VVEAI 已验证的模型 ID `qwen3-8b`，并通过 `extra_body.enable_thinking: false` 适配其非流式接口。未来确定 Qwen3.5 后只需修改 `optimizer.model`。也可以使用安装后的命令 `ragskill-self-optimize`。

## 多任务自优化

`run_task_flow.py` 在同一个 Skill 工作副本上顺序运行多个数据任务。一个 task 可以包含多条问题；框架会先完成该 task 的全部 RAG 执行，再计算 task 汇总指标，并把汇总评估与有限条检索过程交给 Policy Model。每个 task 只优化一次、只修改一个本轮参与的 Skill，后续 task 会继续使用修改后的副本，直到任务列表结束。

默认任务流包含处理好的 HotpotQA、2WikiMultihopQA、TriviaQA validation 和 FinanceBench train 小样本：

```powershell
Copy-Item self_optimization/task_flow.example.yaml self_optimization/task_flow.yaml
python -B run_task_flow.py
```

`self_optimization/task_flow.yaml` 决定 task 顺序、数据路径、单 task 样本数和 RAG 请求参数；Executor、Embedding、Qwen、运行目录和修改边界继续由 `self_optimization/settings.yaml` 管理。安装后也可运行 `ragskill-task-flow`。每次运行的 `task-flow.report.json`、`task-flow.events.jsonl`、Skill 副本、逐 task revision 和 `optimizer_calls/task-XXXX/` 模型审计记录均写入 `self_optimization/runs/<run-id>/`。

Agentic RAG Skill 是一个支持三级 Skill 选择与按需加载的 RAG 研究框架。Executor Model 依次读取 Manage Skill、选择 Agentic RAG Skill、按 workflow 槽位选择 Components Skill，随后将选中的 workflow 与组件编译为可执行 Python 命令。

## Skill 层级

- **Manage Skill**：分析任务并指导 Agentic Skill 选择。
- **Agentic RAG Skill**：定义 Sequential、Parallel 等 RAG 流程，只编排抽象组件槽位。
- **Components Skill**：实现 Retriever、Reranker、Generator 等原子能力。

标准 Skill 包按类型位于 `framework/skills/manage/`、`framework/skills/agentic/` 和 `framework/skills/components/`。每个包以 `SKILL.md` 作为 Agent 框架通用入口，并使用可忽略的 `ragskill.yaml` 与 `scripts/` 扩展本框架运行能力。

## 安装

当前开发环境为 Python 3.13.5：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

只安装基础 framework 时也可以使用：

```powershell
python -m pip install -e .
```

## 配置

仓库只提交无密钥模板。首次运行先创建本地配置：

```powershell
Copy-Item framework/settings.example.yaml framework/settings.yaml
```

在 `framework/settings.yaml` 中配置 Executor API。该文件已加入 `.gitignore`，不得提交真实密钥。模板默认从 `VVEAI_API_KEY` 环境变量读取密钥，也可以仅在被忽略的本地配置中使用 `api_key`。

`skills.root` 指向三个类型目录的共同父目录。它相对配置文件所在的 `framework/` 目录解析，因此当前值为 `skills`，对应 `framework/skills/`；框架会继续从其下的 `manage/`、`agentic/`、`components/` 发现 Skill。

## 运行 Demo

```powershell
python -B run_demo.py
```

入口从 `framework/settings.yaml` 读取 HotpotQA demo 路径、运行条数、请求参数、最终结果路径与中间日志路径，自动完成三级选择、检索、生成及检索/生成测评。检索侧输出 F1@1、Top-n F1（n 为该题 golden 文档数）、MRR、Hit@1、Hit@10、MAP、NDCG、DCG、IDCG；生成侧输出 ChrF、ChrF++、METEOR、R1、R2、RL、PPL、CER、WER，不使用生成 EM/F1。

- 每题预测、单题指标和截至当前题的累计宏平均默认打印到命令行；完整结果写入 `demo.output.result_path`。
- Manage、Agentic、Components、编译、执行和测评事件写入 `demo.output.log_path`。
- `demo.select_skills_per_example: true` 为每题独立选择 Skill；设为 `false` 时整批问题只选择和编译一次并复用。
- 批次选择只发送共享语料统计和均匀抽样的问题文本；`demo.batch_selection_query_sample_size` 控制抽样数，默认 20。
- Vector Retriever 首次运行构建磁盘索引，后续运行直接从 `runtime.vector_index.cache_dir` 加载；缓存由语料、Embedding 配置和文本格式自动失效。
- `python -B run_demo.py --limit 5` 可临时运行 5 条样本。
- 安装后也可使用 `ragskill-demo`。

默认 demo 包含 100 个 HotpotQA 问题和 2000 篇共享文档。原始大型 Parquet 不提交到仓库，已派生的 `corpus.jsonl` 与 `test.jsonl` 会随仓库提供。

## 测试

```powershell
python -B -m pytest -p no:cacheprovider
python -B -m ruff check --no-cache framework tests data/TriviaQA experiments/hotpotqa/scripts experiments/triviaqa/scripts run_demo.py
```

## 目录

```text
.
|-- framework/
|   |-- skills/                 # 真正的 Agent Skills
|   |   |-- manage/             # 高层任务分析与 Agentic Skill 选择
|   |   |-- agentic/            # RAG workflow 与抽象组件槽位
|   |   `-- components/         # Retriever、Generator 等原子实现
|   |-- evaluation/             # XRAG 对齐的检索与生成指标
|   |-- settings.example.yaml   # 可提交配置模板
|   |-- selection.py            # 三级 LLM 选择
|   |-- compiler.py             # workflow 与组件绑定
|   `-- demo.py                 # 配置驱动 demo 入口
|-- experiments/hotpotqa/
|   |-- data/demo/              # 小型可提交 demo 数据
|   `-- scripts/build_demo.py   # 可复现数据构建脚本
|-- data/task_flow/             # 多数据集规范 task JSONL 与来源 manifest
|-- self_optimization/          # 逐样本和多 task Skill 副本自优化
|-- tests/
|-- run_demo.py
|-- run_self_optimization.py
|-- run_task_flow.py
|-- requirements.txt
`-- pyproject.toml
```

更完整的接口和 Skill 包规范见 `framework/SPEC.md`。
