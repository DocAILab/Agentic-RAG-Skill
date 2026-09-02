# 2WikiMultihopQA 数据准备

本目录把 2WikiMultihopQA 的官方 `dev` 分片整理成一个可复现的小型 RAG demo。
构建脚本复用 `experiments/retrieval` 生成的固定 100 题清单，输出共享语料、测试监督和
完整性 manifest；检索字段转换继续由
`experiments/retrieval/adapters/two_wiki.py` 负责，不在此重复实现。

## 数据内容

2WikiMultihopQA 结合 Wikipedia 文本与 Wikidata 关系构造多跳问答。demo 保留：

- `question`、`answer`、`type`；
- 每题的候选文档；
- 文档级和句子级 `supporting_facts`；
- 结构化 `evidences` 推理关系。

测试使用带公开答案和证据标注的 `dev` 分片，不使用缺少公开 gold 的 `test` 分片。

## 环境准备

```powershell
python -m pip install -e ".[experiment,test]"
```

## 固定样本清单

仓库跟踪 `demo/sample_manifest.json`。它由通用检索入口生成：

```powershell
python -m experiments.retrieval.run_manifest --dataset 2wiki --split validation --size 100 --output "data/2WikiMultihopQA/demo/sample_manifest.json"
```

选择规则是稳定 SHA-256 ID 排序，不依赖源数据遍历顺序。清单不是按问题类型强制均分；
构建后的 `manifest.json` 会记录实际类型分布。

## 构建 demo

在仓库根目录执行：

```powershell
python -B data/2WikiMultihopQA/build_demo.py
```

默认输出：

```text
data/2WikiMultihopQA/demo/
|-- sample_manifest.json    # 跟踪：固定问题 ID
|-- corpus.jsonl            # 本地：去重后的共享文档库
|-- test.jsonl              # 本地：问题、答案、证据与候选文档 ID
`-- manifest.json           # 本地：来源、数量、类型分布和文件 SHA-256
```

生成文件默认不提交 Git。相同输入重复执行会直接复用相同内容；如果输出内容不同，脚本会
拒绝覆盖。确认需要重建时使用：

```powershell
python -B data/2WikiMultihopQA/build_demo.py --force
```

## 固定来源

- Hugging Face 数据集：`xanhho/2WikiMultihopQA`
- revision：`612bc5039a457880d9e7d84c3b0a4cf154b70e4f`
- split：`dev`
- license：Apache-2.0

固定 revision 与样本清单 digest 会写入本地 `manifest.json`，用于复现实验。

## 输出契约

`corpus.jsonl` 每行包含：

```text
id, title, text, sentences, source_question_ids
```

`test.jsonl` 每行包含：

```text
id, question, answer, answers, type, answer_type,
relevant_document_ids, supporting_facts, candidate_document_ids, evidences
```

共享文档 ID 由“标题 + 内容 SHA-256 前缀”组成，同名但正文不同的 Wikipedia 快照不会互相
覆盖；同一题内完全重复的标题使用 `#2` 后缀。构建阶段会拒绝空问题、空答案、空语料、
重复问题 ID、缺失证据文档以及越界的证据句子编号。

## 验证

```powershell
python -B -m pytest tests/test_2wikimultihopqa_demo.py -q
python -B -m ruff check --no-cache data/2WikiMultihopQA tests/test_2wikimultihopqa_demo.py
```

提交前确认本地生成文件未进入暂存区：

```powershell
git status --short
git check-ignore data/2WikiMultihopQA/demo/corpus.jsonl
git check-ignore data/2WikiMultihopQA/demo/test.jsonl
git check-ignore data/2WikiMultihopQA/demo/manifest.json
```
