# Retrieval Benchmark

该入口直接评测 BM25 或 BGE-Large Component，不经过 Manage/Agentic/Generator，也不会调用 DeepSeek。每个问题只在数据集提供的候选文档中检索。

## 安装实验依赖

```powershell
python -m pip install -e ".[experiment,embedding]"
```

## 运行

HotpotQA BM25：

```powershell
python -m experiments.retrieval.run_benchmark `
  --dataset hotpotqa `
  --split validation `
  --retriever bm25
```

HotpotQA BGE-Large：

```powershell
python -m experiments.retrieval.run_benchmark `
  --dataset hotpotqa `
  --split validation `
  --retriever vector `
  --model BAAI/bge-large-en-v1.5 `
  --device cuda `
  --batch-size 16
```

数据集名称还支持 `2wiki` 和 `triviaqa`。2Wiki 的 `validation` 自动映射到官方 `dev` split。可用 `--limit 10` 先做烟雾测试；显存不足时减小 `--batch-size` 或使用 `--device cpu`。

真实 BGE 小样本集成测试默认跳过，显式启用方式：

```powershell
$env:RAGSKILL_RUN_BGE_INTEGRATION = "1"
$env:RAGSKILL_BGE_DEVICE = "cuda"  # 没有 GPU 时改为 cpu
python -m pytest tests/test_retrieval_bge_integration.py
```

## 输出与恢复

默认输出目录为 `experiments/retrieval/outputs/<dataset>/<retriever>/`：

- `results.jsonl`：逐样本追加的检索结果、标签和指标；
- `checkpoint.json`：最近进度与运行参数；
- `summary.json`：计数和按标签类型分组的宏平均。

使用相同参数重新执行时会读取 `results.jsonl` 并跳过已经完成的样本。参数不一致时会拒绝复用旧结果，防止混合实验。HotpotQA 和 2Wiki 使用 `supporting_facts` 强标签；TriviaQA 使用 `weak_answer_alias` 弱标签，二者不会合并汇总。无公开 gold 的样本只保存检索结果，`metrics` 为 `null`。
