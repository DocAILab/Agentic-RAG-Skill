# FinanceBench Demo Data

该目录由 `python -m experiments.financebench.scripts.build_demo` 生成。`corpus.jsonl`、
`test.jsonl` 和 `manifest.json` 属于派生数据，不手工编辑。

语料仅覆盖公开样本中出现的标注证据页，不是原始 PDF 的完整页集合。每个测试问题使用固定
10 个候选页，包含全部标注证据页和按同财报、同公司、其他财报优先级确定性选择的负例。
