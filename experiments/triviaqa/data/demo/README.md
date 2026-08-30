# TriviaQA Demo Dataset

该目录是从 `data/TriviaQA` 标准化子集确定性派生的小型 demo 数据，适合验证
framework 的检索、生成与测评链路，不用于报告正式 benchmark 结果。

## 文件

- `corpus.jsonl`：共享小语料库，每行包含 `id`、`title`、`text`、`source` 与来源问题 ID。
- `test.jsonl`：20 条测试问题，每行包含题目、答案与别名、候选文档 ID 和弱标签相关
  文档 ID（答案别名出现在文档标题或正文中）。
- `manifest.json`：子集文件校验值、抽样规则、样本数量和两个 JSONL 文件的 SHA-256。

## 重建

```powershell
python -B "data/TriviaQA/加载脚本.py" --max-query-samples 100
python -B experiments/triviaqa/scripts/build_demo.py --input data/TriviaQA/outputs/wikipedia-dev_subset_100.json
```

## 运行 Demo

```powershell
$env:DEEPSEEK_API_KEY = "..."
python -B run_demo.py --config experiments/triviaqa/configs/demo.example.yaml --limit 20
```

本地配置固定为 BM25 检索（不加载 embedding 模型，跑得快）。向量化路径用
`experiments/triviaqa/scripts/build_vectors.py` 单独验证；服务器上如需用
Vector Retriever，把 `retriever` 改为 `component-vector-retriever` 并开启
`embedding`（`device: cuda`），注意整篇长文档超过 bge 512 token 上限，需先切块。
