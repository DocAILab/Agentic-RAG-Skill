# HotpotQA Demo Dataset

该目录是从本地只读 `distractor/validation` 分片确定性派生的小型 demo 数据，适合验证 framework 的检索、生成与测评链路，不用于报告正式 benchmark 结果。

## 文件

- `corpus.jsonl`：共享小语料库，每行包含 `id`、`title`、`text`、原始句子和来源问题 ID。
- `test.jsonl`：20 条测试问题，每行包含答案、问题类型、候选文档 ID、相关文档 ID 和 supporting facts。
- `manifest.json`：源文件校验值、采样规则、样本数量和两个 JSONL 文件的 SHA-256。

测试集固定包含 10 条 bridge 与 10 条 comparison 问题；comparison 中包含 2 条 yes、2 条 no 和 6 条普通 span 答案。语料库是这 20 条问题各自 10 个 distractor context 的去重并集。

## 重建

```powershell
python -B experiments/hotpotqa/scripts/build_demo.py
```

构建脚本只读取 `data/raw/`，所有派生文件均写入当前 `data/demo/` 目录。

## 运行 Demo

```powershell
python -B run_demo.py
```

入口从 `framework/settings.yaml` 的 `demo` 段读取 corpus、测试集、样本数量、请求参数和结果路径，自动完成三级 Skill 选择、检索生成、Hit@1/Hit@10/EM/F1 测评及结果保存。安装项目后也可以直接运行 `ragskill-demo`。

默认配置只运行第一题，并使用该题原始 10 篇 distractor 文档和 BM25。常用配置调整：

- `max_examples: 20`：运行完整 demo；`null` 表示运行测试文件中的全部样本。
- `candidate_documents_only: false`：在共享的 200 篇小语料中检索。
- 删除 `demo.request.constraints`：允许模型自适应选择 BM25 或 Vector Retriever。
- `demo.output.result_path`：设置逐题记录与宏平均指标的最终 JSON 输出位置。
- `demo.output.log_path`：设置三级选择、执行 trace、答案和逐题指标的 JSONL 中间日志路径。

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
