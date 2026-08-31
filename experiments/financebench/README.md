# FinanceBench Experiments

本目录将 `data/FinanceBench/` 的确定性本地版本转换为 framework 和 retrieval benchmark
共用的 demo 契约。

## 语料范围

当前公开数据只包含 150 条标注样本及其证据页，没有完整 PDF 页级语料。构建脚本会聚合所有
`evidence_text_full_page`，生成共享的 **closed evidence-page corpus**。每个问题固定使用
10 个候选页：保留全部 gold 页，负例依次优先同一财报、同一公司和其他财报页面，并使用稳定
SHA-256 排序保证可复现。

这种设置可以比较检索器，但候选页都曾被选为某道题的 gold evidence，难度和完整 PDF 检索
不同。结果不得表述为完整 FinanceBench PDF retrieval 成绩。后续取得全量数据或解析全部 PDF
后，应增加新的 corpus scope，而不是覆盖当前契约。

## 构建

```powershell
python data/FinanceBench/加载脚本.py --version small
python -m experiments.financebench.scripts.build_demo
```

可用 `--limit 20` 确定性选择部分测试问题；共享 corpus 仍由完整 small 版本构建，保证存在负例。

生成文件位于 `experiments/financebench/data/demo/`：

- `corpus.jsonl`：按 `doc_name#p<page>` 去重的共享证据页；
- `test.jsonl`：问题、答案、固定 10 个候选页 ID 和 gold 相关页 ID；
- `manifest.json`：固定 revision、输入/输出 SHA-256、语料范围和抽样参数。

## 检索评测

```powershell
python -m experiments.retrieval.run_benchmark `
  --dataset financebench `
  --retriever bm25
```

FinanceBench 默认使用上述本地 demo，逻辑 split 为 `test`。使用其他目录时传入
`--data-dir <demo-directory>`。

## 端到端 RAG demo

配置示例复用同一份 corpus/test 契约，并固定为 BM25 + Iterative RAG：

```powershell
$env:DEEPSEEK_API_KEY = "..."
python run_demo.py --config experiments/financebench/configs/demo.example.yaml
```

该命令会调用配置的生成模型；上一节的 retrieval benchmark 不调用 LLM。

数据集 revision：`e04404e3a97f69f79c14d42f24981a1c9c3bcd18`。
许可证：CC BY-NC 4.0。
