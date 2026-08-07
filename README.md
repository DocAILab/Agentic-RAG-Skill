# Agentic RAG Skill

Agentic RAG Skill 是一个支持三级 Skill 选择与按需加载的 RAG 研究框架。Executor Model 依次读取 Manage Skill、选择 Agentic RAG Skill、按 workflow 槽位选择 Components Skill，随后将选中的 workflow 与组件编译为可执行 Python 命令。

## Skill 层级

- **Manage Skill**：分析任务并指导 Agentic Skill 选择。
- **Agentic RAG Skill**：定义 Sequential、Parallel 等 RAG 流程，只编排抽象组件槽位。
- **Components Skill**：实现 Retriever、Reranker、Generator 等原子能力。

标准 Skill 包位于 `framework/skills/`。每个包以 `SKILL.md` 作为 Agent 框架通用入口，并使用可忽略的 `ragskill.yaml` 与 `scripts/` 扩展本框架运行能力。

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

`skills.root` 相对配置文件所在的 `framework/` 目录解析，因此当前值为 `skills`，对应 `framework/skills/`。

## 运行 Demo

```powershell
python -B run_demo.py
```

入口从 `framework/settings.yaml` 读取 HotpotQA demo 路径、运行条数、请求参数、最终结果路径与中间日志路径，自动完成三级选择、检索、生成和 Hit@1、Hit@10、EM、F1 测评。

- 最终结果默认打印到命令行，并写入 `demo.output.result_path`。
- Manage、Agentic、Components、编译、执行和测评事件写入 `demo.output.log_path`。
- `python -B run_demo.py --limit 5` 可临时运行 5 条样本。
- 安装后也可使用 `ragskill-demo`。

默认 demo 包含 20 个 HotpotQA 问题和 200 篇共享文档。原始大型 Parquet 不提交到仓库，已派生的小型 `corpus.jsonl` 与 `test.jsonl` 会随仓库提供。

## 测试

```powershell
python -B -m pytest -p no:cacheprovider
python -B -m ruff check --no-cache framework tests experiments/hotpotqa/scripts run_demo.py
```

## 目录

```text
.
|-- framework/
|   |-- skills/                 # Manage、Agentic 与 Components Skill 包
|   |-- evaluation/             # Hit@K、EM、F1
|   |-- settings.example.yaml   # 可提交配置模板
|   |-- selection.py            # 三级 LLM 选择
|   |-- compiler.py             # workflow 与组件绑定
|   `-- demo.py                 # 配置驱动 demo 入口
|-- experiments/hotpotqa/
|   |-- data/demo/              # 小型可提交 demo 数据
|   `-- scripts/build_demo.py   # 可复现数据构建脚本
|-- tests/
|-- run_demo.py
|-- requirements.txt
`-- pyproject.toml
```

更完整的接口和 Skill 包规范见 `framework/SPEC.md`。
