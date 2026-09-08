# TriviaQA 数据准备

本目录提供 TriviaQA 的统一本地数据入口。加载脚本从本地 RC 原始数据生成标准化文本子集：

- `wikipedia-dev_subset_100.json`、`wikipedia-dev_subset_800.json`：默认生成的两档子集，
  规模通过 `--max-query-samples` 指定。
- 子集按 `SHA-256("{seed}:{QuestionId}")` 确定性排序后取前缀，`subset_100 ⊆ subset_800`，
  可逐档累加构建索引。

原始数据与派生子集仅保存在本地，不提交到 Git。仓库只跟踪加载代码、说明和测试。

## 环境准备

仅依赖 Python 标准库（3.11+），无需第三方包。原始数据需自行下载：

1. 下载 RC 包（约 2.5 GB）：<http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz>
2. 解压后把 `qa/` 与 `evidence/` 放到 `data/raw/triviaqa/` 下，问答文件建议使用
   `wikipedia-dev.json`（`test` 分片无答案，会被全部过滤）。

```text
data/raw/triviaqa/
|-- qa/
|   `-- wikipedia-dev.json
`-- evidence/
    |-- wikipedia/    # 维基证据，纯文本 <标题>.txt
    `-- web/          # 网页证据，数字子目录 <id>/<id>_*.txt
```

> Windows 注意：约 455 个维基证据文件名含 `:`、`?` 等非法字符，解压时替换为下划线；
> 脚本会先按原始文件名查找，找不到时自动回退到清洗后的文件名。

## 生成数据

在项目根目录执行：

```powershell
# 默认生成 100 和 800 两档子集
python -B "data/TriviaQA/prepare_triviaqa.py"

# 指定其他问答文件与多档规模
python -B "data/TriviaQA/prepare_triviaqa.py" --source data/raw/triviaqa/qa/web-dev.json --max-query-samples 100 800 5000
```

默认输出到 `data/TriviaQA/outputs/`。使用 `--output PATH` 更换输出目录，`--seed` 更换抽样种子。

脚本重复执行时会核对 `outputs/manifest.json` 中的文件大小、记录数和 SHA-256，有效子集直接
复用；数据缺失、损坏或参数不一致时脚本拒绝覆盖。确认需要重建后使用：

```powershell
python -B "data/TriviaQA/prepare_triviaqa.py" --force
```

## 本地目录

```text
data/TriviaQA/
|-- README.md
|-- prepare_triviaqa.py
|-- triviaqa_data.py
|-- _manifest.py
`-- outputs/                  # git 忽略，仅本地
    |-- manifest.json
    |-- wikipedia-dev_subset_100.json
    `-- wikipedia-dev_subset_800.json
```

## 可复现规则

- 数据源：本地 TriviaQA RC 包，默认 `qa/wikipedia-dev.json`，数据源指纹为该文件 SHA-256。
- 抽样：按 `SHA-256("{seed}:{QuestionId}")` 排序取前缀，默认 seed `20260828`。
- 子集关系：`subset_100 ⊆ subset_800`，可逐档累加构建索引。
- 过滤：空 `QuestionId`/`Question`/`Answer.Value`、解析不到证据文档、重复 ID 会被过滤，
  统计写入每个子集 JSON 的 `sampling.filtered`。
- 保存格式：UTF-8 JSON；样本字段为 `sample_id`、`query`、`golden_answer`、
  `golden_answers`、`documents`（`{id, title, text, source}`）。

## 加载示例

```python
import json

with open("data/TriviaQA/outputs/wikipedia-dev_subset_100.json", encoding="utf-8") as handle:
    payload = json.load(handle)

print(payload["counts"]["samples"])          # 100
print(payload["samples"][0].keys())
```

向量化使用 `experiments/triviaqa/scripts/build_vectors.py`；检索基准
（`experiments/retrieval`，已支持 `--dataset triviaqa`）从 Hugging Face 原始数据流式读取，
不消费本目录子集，不要在本目录重复实现字段转换。

## 提交检查

提交前确认 `outputs/`、原始数据与向量库没有进入暂存区：

```powershell
git status --short
git diff --cached --stat
git check-ignore data/TriviaQA/outputs
```
