# RAGSkill Framework Specification V0.1

本规范定义 SkillAdaptive 的三类 Skill 及其跨 Agent framework 兼容边界。四阶段分级选择、workflow 绑定、编译命令和端到端执行均已实现。

## 1. 四阶段执行语义

1. **Load Manage**：只向 LLM 暴露一个 Manage Skill 的完整 `SKILL.md`，不暴露 Agentic 或 Component 正文。
2. **Select Agentic**：按照 Manage Skill 指导，仅广告 Agentic RAG Skill 的 `name + description`；LLM 选择一个后才加载其正文与 `ragskill.yaml`。
3. **Select Components**：读取选中 Agentic Skill 的槽位和指导；每个槽位只广告接口兼容的 Component Skill，选中后才加载其正文和脚本。
4. **Compile and Run**：生成一个 Python 指令。该指令加载选中 Agentic 工作流脚本，将槽位绑定到选中 Component 实现，然后执行检索与生成。

任何阶段都不得提前向 LLM 暴露下一层全部正文。

## 2. 模型接口与当前入口

framework 只依赖统一的 `ModelClient.generate()`，具体模型服务通过以下适配器接入：

- `OpenAICompatibleModelClient`：调用 `/v1/chat/completions`，支持 OpenAI 官方 API、vLLM、SGLang、Ollama 等兼容服务。
- `AnthropicModelClient`：调用 `/v1/messages`，使用 `x-api-key` 和 `anthropic-version`。
- `OpenAICompatibleEmbeddingClient`：调用 `/v1/embeddings`，通过独立 `EmbeddingClient` 注入 Vector Retriever。
- `SentenceTransformerEmbeddingClient`：延迟加载本地 sentence-transformers 权重，不选择 Vector Retriever 时不会下载或载入模型。

两个远程 Model 客户端和 OpenAI-compatible Embedding 客户端均使用 Python 标准库发送 HTTP 请求，不要求 Skill 依赖厂商 SDK。API key 默认分别读取 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY`；本地 sentence-transformers 客户端需要安装项目的 `embedding` 可选依赖。

OpenAI-compatible 示例：

```python
from framework import create_model_client, select_rag_plan

model = create_model_client(
    "openai-compatible",
    model="executor-model",
    base_url="http://localhost:8000/v1",
)
plan = select_rag_plan(request, model=model, skill_root="framework/skills")
```

Anthropic 示例：

```python
from framework import create_model_client, select_rag_plan

model = create_model_client("anthropic", model="claude-model-name")
plan = select_rag_plan(request, model=model, skill_root="framework/skills")
```

`select_rag_plan()` 是选择入口，会真实调用模型三次：生成 Manage 指导、选择一个 Agentic Skill、按 Agentic 槽位选择 Components。返回值是经过契约校验的 `RAGSelectionPlan`。

前三步也提供独立入口，便于单独训练、评测和检查模型上下文：

```python
from framework import (
    run_manage_stage,
    select_agentic_skill,
    select_component_skills,
)

manage_result = run_manage_stage(
    request,
    model=model,
    skill_root="framework/skills",
)
agentic_result = select_agentic_skill(
    request,
    manage_result=manage_result,
    model=model,
    skill_root="framework/skills",
)
component_result = select_component_skills(
    request,
    agentic_result=agentic_result,
    model=model,
    skill_root="framework/skills",
)
```

`select_agentic_skill()` 的模型提示仅包含 Manage 指导、RAG 请求和全部 Agentic `name + description`。模型返回的名称通过候选集合与 `kind=agentic` 双重校验，之后才读取被选中 Skill 的完整 `SKILL.md`，保存在 `AgenticStageResult.instructions` 中供第三步使用。

`select_component_skills()` 只广告与 Agentic 槽位 capability、输入类型和输出类型一致的 Component `name + description`。模型输出还必须通过槽位集合、数量、唯一性和兼容性校验；之后才读取选中 Component 的正文。

三级选择调用默认为推理模型预留最多 8192 个输出 token。Executor、Generator 与其他 LLM Component 的配置默认值同样不低于 8192；调用方显式传入合法 `max_tokens` 时保留该值，包括小于 8192 的值。

HTTP 模型客户端默认对临时网络错误、408、429 和 5xx（包括 524）额外重试 2 次，并以 2 秒为初值执行指数退避。`executor.options.max_retries` 和 `executor.options.retry_backoff_seconds` 可覆盖这两个值；400 等确定性请求错误不会重试。`timeout_seconds` 只控制本地等待时间，不能延长第三方 API 网关自身的上游等待上限。

Vector Retriever 的索引生命周期与 XRAG 对齐：首次遇到某套共享语料时构建归一化稠密向量矩阵，并在 `runtime.vector_index.cache_dir` 下持久化为 `manifest.json + vectors.npy`；后续问题复用内存索引，后续 Python 进程直接加载磁盘索引，每题只编码 query。缓存键由有序文档 ID 与正文、Embedding provider/model/base URL/options 和 `title+text` 格式版本共同计算；任一输入变化都会进入新目录，不会误用旧向量。`manifest.json` 不保存语料正文，索引临时文件通过原子替换发布，损坏或校验失败时自动重建。

Vector Component 优先调用可选的 `context.search_vector_index()`，在 Claude Code 等只实现基础 `context.embed()` 的环境中仍可退回直接余弦检索。索引的 `builds`、`disk_loads`、`memory_hits`、最近来源与路径写入 `vector_index_cache`；每个 Component 的名称、槽位、状态和 `duration_seconds` 写入 `component_timings`，用于区分索引构建、检索、重排和生成延迟。

端到端入口为：

```python
from framework import run_rag

result = run_rag(
    request,
    model=model,
    embedding_model=embedding_model,
    skill_root="framework/skills",
)
```

`run_rag()` 在选择完成后只加载被选中的 Agentic 和 Component Python 脚本，构造 `RuntimeComponentContext`，编译槽位绑定并执行检索与生成。`embedding_model` 仅在选中的 Component 需要向量时必需。

### 配置文件入口

framework 运行时统一读取同目录下的本地 `settings.yaml`。配置包含 Skill 根目录、Manage Skill、请求默认值、Executor API、可选远程或本地 Embedding、超时、额外请求头和 provider 专用参数。`api_key` 可供本地忽略文件直写密钥，`api_key_env` 用于按环境变量名加载密钥，两者互斥。仓库只提交不含真实密钥的 `settings.example.yaml`。

```python
from framework import run_rag_from_config

result = run_rag_from_config(
    request,
    config_path="framework/settings.yaml",
)
```

相对 `skills.root` 以配置文件所在目录为基准解析。`runtime.request_defaults` 为请求提供 `top_k`、`max_tokens`、`rank_constant` 等默认值，调用参数优先覆盖配置。`embedding.enabled: false` 会关闭向量客户端；若后续选择 Vector Retriever，执行器将给出缺失 Embedding 服务的明确错误。

当前本地配置使用 VVEAI OpenAI-compatible API 的 `deepseek-v4-flash` 作为 Executor，并使用 `BAAI/bge-large-en-v1.5` 作为本地 Embedding。VVEAI 当前账户对远程模型名 `bge-large-v1.5-en` 没有可用 Embeddings 渠道，因此不把该远程调用保留为默认配置。

### Demo 入口

`settings.yaml` 的 `demo` 段统一声明 `corpus_path`、`test_path`、`output.result_path`、`output.log_path`、`max_examples`、`candidate_documents_only`、`select_skills_per_example`、`batch_selection_query_sample_size` 和请求覆盖参数。用户不需要手工加载 JSONL 或拼接 framework API：

```powershell
python -B run_demo.py
```

`framework.demo.run_demo()` 只创建一次模型客户端。`select_skills_per_example: true` 时，每题独立执行三级选择和编译，用于 query-level adaptive RAG；设为 `false` 时，framework 根据请求参数、共享语料统计和均匀抽样的问题文本只选择并编译一次，所有问题复用同一命令。问题抽样数由 `batch_selection_query_sample_size` 控制，默认 20；框架同时发送总问题数与实际抽样数，不发送完整问题集。两种模式都会逐题执行检索、生成和 Hit@1、Hit@10、EM、F1 测评。终端在每题后打印单题指标与截至当前题的累计宏平均；最终答案、Skill 选择、检索 ID、trace、编译指令与整体宏平均写入 `output.result_path`。所有阶段事件以带 `run_id` 的 JSON Lines 追加到 `output.log_path`。安装项目后等价命令为 `ragskill-demo`；`--limit N` 可临时覆盖运行条数。

## 3. 可移植 Skill 包

Skill 根目录必须先按职责分为三级，类型目录本身不是 Skill 包：

```text
skills/
├── manage/
│   └── <manage-skill-name>/
├── agentic/
│   └── <agentic-skill-name>/
└── components/
    └── <component-skill-name>/
```

`ragskill.yaml.kind` 必须与所在类型目录一致：`manage/` 对应 `manage`，`agentic/` 对应 `agentic`，`components/` 对应 `component`。框架拒绝根目录平铺、未知层级、额外嵌套和类型错放的 Skill，避免选择阶段跨层加载。

每个类型目录中的 Skill 都必须首先是标准 Agent Skill：

```text
<skill-name>/
├── SKILL.md               # 标准入口；只使用 name 和 description
├── agents/
│   └── openai.yaml        # 可选的 Codex UI 元数据
├── ragskill.yaml          # SkillAdaptive 可选扩展；其他框架可忽略
└── scripts/
    ├── workflow.py        # Agentic Skill 使用
    └── component.py       # Component Skill 使用
```

Claude Code、Codex 和其他 Agent Skills consumer 直接读取 `SKILL.md`。SkillAdaptive 额外读取 `ragskill.yaml` 和 Python 脚本。不得把三层类型、运行入口或槽位放进私有 `SKILL.md` frontmatter。

## 4. Manage Skill

Manage Skill 是纯选择指导，不执行 RAG，也不声明 Python runtime。

```yaml
schema_version: 1
runtime_id: manage.rag.default
kind: manage
version: 0.1.0
mutable: false
selection:
  target_kind: agentic
  min: 1
  max: 1
```

其 `SKILL.md` 必须说明如何根据 query、任务、数据与预算生成 Agentic RAG Skill 选择指导，但不能直接选择 Component Skill。

## 5. Agentic RAG Skill

Agentic Skill 本质是流程安排。它必须包含 `scripts/workflow.py`，但不得导入或实现具体 Retriever、Reranker、Generator。

```yaml
schema_version: 1
runtime_id: agentic.sequential.vanilla_rag
kind: agentic
runtime:
  type: python-workflow
  path: scripts/workflow.py
  callable: run
slots:
  rewriter:
    capability: rewriter
    input: RewriteRequest
    output: RewriteResult
    min: 0
    max: 1

  retriever:
    capability: retriever
    input: RetrievalRequest
    output: RetrievalResult
    min: 1
    max: 1
```

统一入口：

```python
def run(request, components):
    """通过抽象组件槽位执行 Agentic workflow。"""
    retrieval = components.call("retriever", request)
    return components.call("generator", retrieval)
```

`components` 只提供：

```python
components.has(slot)
components.call(slot, inputs, index=0)
components.call_all(slot, inputs)
```

Agentic 脚本可以实现条件、循环、并行、融合、重试和终止逻辑，但所有原子 RAG 能力必须通过槽位空调用完成。

当前 Vanilla RAG 的可选 `rewriter` 槽位采用 single-sample HyDE（`N=1`）：Rewriter 接收原始查询，只生成一个假设文档，并将单个 `rewritten_query` 交给语义 Vector Retriever。Retriever 之后的 Reranker 和 Generator 继续使用原始查询；假设文档不得加入检索证据或直接用于最终回答。

## 6. Component Skill

Component Skill 必须提供具体原子实现，并声明一个或多个 capability。

```yaml
schema_version: 1
runtime_id: component.retrieval.bm25
kind: component
runtime:
  type: python-component
  path: scripts/component.py
  callable: run
provides:
  retriever:
    input: RetrievalRequest
    output: RetrievalResult
```

统一入口：

```python
def run(inputs, context):
    """使用注入的运行上下文执行具体原子组件。"""
    ...
    return outputs
```

`context` 提供冻结 Executor Model、Embedding Model 等外部服务。Component 不选择其他 Skill，不安排跨组件流程。

Component 可以通过可选的 `requires` 声明跨 capability 的组件兼容要求。每个 key 是所需 capability，每个 `components` 列表给出允许绑定的 Component 包名：

```yaml
provides:
  rewriter:
    input: RewriteRequest
    output: RewriteResult
requires:
  retriever:
    components:
      - component-vector-retriever
```

Skill 发现阶段校验被引用的 Component 存在并且确实提供对应 capability；Component 选择阶段和编译阶段都会再次校验实际绑定。single-sample HyDE 使用该声明强制要求 Vector Retriever，因此 `HyDE + BM25` 会被拒绝，即使模型或调用方显式返回了该组合。

## 7. 标准数据包络

V0 样例使用 JSON-compatible dictionary：

- `RewriteRequest`：`query`, 可选 `temperature`, 可选 `max_tokens`
- `RewriteResult`：`rewritten_query`
- `RetrievalRequest`：`query`, `documents`, `top_k`
- `RetrievalResult`：`documents`
- `RerankRequest`：`query`, `documents`, `top_k`
- `RerankResult`：`documents`
- `GenerationRequest`：`query`, `documents`, `max_tokens`
- `GenerationResult`：`answer`
- `RAGRequest`：`query`, `documents`, `top_k`, `max_tokens`, 可选 `rewrite_temperature`, 可选 `rewrite_max_tokens`
- `RAGResult`：`answer`, `documents`, `trace`

`RewriteRequest.query` 必须是非空原始查询。`temperature` 是可选的非负生成温度，默认值为 `0.0`；`max_tokens` 是可选的正整数生成长度上限，默认值为 `8192`。`RewriteResult.rewritten_query` 必须是非空字符串，并且只包含一个假设答案式文档。

V0 的 HyDE 接口固定为 single-sample HyDE（`N=1`）：每个查询只调用一次生成模型，不返回假设文档列表，也不在 Rewriter 中执行多样本向量平均。`rewritten_query` 只作为 Vector Retriever 的检索查询；原始 `query` 保留给 Reranker 和 Generator。

后续版本可以将这些包络升级为 JSON Schema，但接口名称和责任边界必须保持稳定。

## 8. 编译与执行

`compile_rag_command()` 将 `RAGSelectionPlan` 编译为可调用的 `CompiledRAGCommand`。其 `instruction` 属性生成等价于以下结构的一条 Python 指令：

```python
run_compiled_rag(
    workflow="agentic-vanilla-rag",
    bindings={
        "rewriter": [],
        "retriever": ["component-bm25-retriever"],
        "reranker": [],
        "generator": ["component-grounded-generator"],
    },
    request=request,
    skill_root=skill_root,
    context=context,
)
```

`BoundComponentInvoker` 将 workflow 中的 `components.call(...)` 和 `components.call_all(...)` 转发到槽位绑定的具体实现。`RuntimeComponentContext` 将冻结 Executor Model 和可选 Embedding Model 注入 Component。

编译产物只记录选中的包和槽位绑定，不复制 Skill 源码，也不修改任何基础 Skill。`run_compiled_rag()` 用于复现给定绑定，`run_rag()` 用于一次调用完成选择、编译和执行。

## 9. 测评

首版测评模块位于 `framework/evaluation/`，明确区分检索质量与答案质量：

- `Hit@1`、`Hit@10`：若前 K 个检索文档中至少出现一个相关文档标识符，则该样本得 1，否则得 0。
- `EM`：预测答案与标准答案经过小写化、移除英文标点和冠词、合并空白后完全一致，则得 1。
- `F1`：对归一化后的预测答案与标准答案计算词元级精确率和召回率的调和平均。
- 一条样本存在多个标准答案别名时，`EM` 和 `F1` 分别取别名中的最高分；批量入口对各样本做宏平均。

这里的 `EM/F1` 是 HotpotQA 风格的**答案指标**。只读参考项目 XRAG 曾使用同名指标比较检索 ID 集合，新 framework 不沿用该混名。

```python
from framework import evaluate_rag_result

metrics = evaluate_rag_result(
    result,
    gold_answers=["Paris", "Paris, France"],
    relevant_ids={"support-document-id"},
)
print(metrics.to_dict())
```

`evaluate_example()` 接收显式 `EvaluationExample`，适合构造离线样本；`evaluate_batch()` 返回样本数和四项指标的宏平均；`evaluate_rag_result()` 可直接消费 `run_rag()` 的 `answer` 与有序 `documents`，默认从每个文档的 `id` 字段读取检索标识符。
