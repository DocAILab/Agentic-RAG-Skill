# HotpotQA Demo Dataset

该目录是从本地只读 `distractor/validation` 分片确定性派生的小型 demo 数据，适合验证 framework 的检索、生成与测评链路，不用于报告正式 benchmark 结果。

## 文件

- `corpus.jsonl`：共享小语料库，每行包含 `id`、`title`、`text`、原始句子和来源问题 ID。
- `test.jsonl`：100 条测试问题，每行包含答案、问题类型、候选文档 ID、相关文档 ID 和 supporting facts。
- `manifest.json`：源文件校验值、采样规则、样本数量和两个 JSONL 文件的 SHA-256。

测试集固定包含 50 条 bridge 与 50 条 comparison 问题；comparison 中包含 10 条 yes、10 条 no 和 30 条普通 span 答案。语料库先收录这 100 条问题各自的 10 个 distractor context，再从同一 validation split 按固定哈希顺序补充背景文档，最终得到 2000 篇去重文档。

## 重建

```powershell
python -B experiments/hotpotqa/scripts/build_demo.py
```

构建脚本只读取 `data/raw/`，所有派生文件均写入当前 `data/demo/` 目录。

## 运行 Demo

```powershell
python -B run_demo.py
```

入口从 `framework/settings.yaml` 的 `demo` 段读取 corpus、测试集、样本数量、请求参数和结果路径，自动完成三级 Skill 选择、检索生成、XRAG 对齐的八项检索指标和九项生成指标测评及结果保存。安装项目后也可以直接运行 `ragskill-demo`。

默认配置运行全部 100 题，在共享的 2000 篇语料上允许模型自适应选择 Retriever。常用配置调整：

- `max_examples: 1`：只做一题连通测试；`100` 或 `null` 运行完整 demo。
- `candidate_documents_only: true`：只在每题原始 10 篇 distractor 中检索。
- `select_skills_per_example: true`：每题独立选择 Skill；设为 `false` 时整批测试只选择一次并复用。
- `batch_selection_query_sample_size: 20`：批次只选择一次时，向模型均匀抽样发送的问题文本数量；语料仅发送统计信息。
- `runtime.vector_index.cache_dir`：Vector Retriever 的磁盘索引目录；首次构建后，后续运行只编码 query。
- `demo.output.result_path`：设置逐题记录与宏平均指标的最终 JSON 输出位置。
- `demo.output.log_path`：设置三级选择、执行 trace、答案和逐题指标的 JSONL 中间日志路径。

当前默认配置不限制 Retriever 类型，模型可以在兼容的 Component Skill 中自适应选择。终端每完成一题都会同时打印该题指标和当前已完成样本的累计宏平均。

临时覆盖运行条数而不修改 YAML：

```powershell
python -B run_demo.py --limit 5
```

## Optimized SIM-RAG rerun

The tracked configuration fixes the same 20-example candidate-document subset
and requests `agentic-sim-rag` with BM25, the grounded Generator, and the
Critic. It keeps answer generation at 256 tokens while giving the Critic an
independent 4096-token budget:

```powershell
python -B -m experiments.hotpotqa.scripts.run_sim_rag --config experiments/hotpotqa/configs/sim_rag_optimized.example.yaml
```

Set `DEEPSEEK_API_KEY` before running. The schema-v2 report includes Hit@1,
Hit@10, Recall@10, All-Support@10, EM, F1, and per-iteration support gain.

## Adaptive SIM-RAG Component selection

This runner fixes `agentic-sim-rag` but asks the Executor Model to select its
compatible Components independently for every question. It records the selected
bindings, selection reason, aggregate selection counts, execution trace, and
metrics in a schema-v3 report:

```powershell
python -B -m experiments.hotpotqa.scripts.run_sim_rag_adaptive --config experiments/hotpotqa/configs/sim_rag_adaptive.example.yaml
```

Before a real run, install the optional local model dependency with
`pip install -e ".[embedding,rerank]"`. The first Vector or BGE selection may
download its configured model weights. The fixed-BM25 runner remains the control
baseline for comparing adaptive selection.
