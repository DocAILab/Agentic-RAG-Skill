# HotpotQA 数据准备

本目录提供 HotpotQA 的统一本地数据入口。加载脚本会实际生成两个互相独立的数据版本：

- `full/`：官方 `distractor` 和 `fullwiki` 配置的全部可用 split。
- `small/`：从 `distractor/train` 固定选出的 5,000 条训练样本。

两个数据目录仅保存在本地，不提交到 Git。仓库只跟踪加载代码、说明和测试。

## 环境准备

使用 Python 3.11 至 3.13。项目固定依赖目前不支持 Python 3.14。

```powershell
py -3.12 -m pip install -r requirements.txt
```

完整数据的 Parquet 文件约 747 MB。Hugging Face 下载缓存还会占用额外空间，建议至少预留 2 GB。

## 生成数据

在项目根目录执行：

```powershell
# 同时生成完整版本和 5,000 条小版本
py -3.12 data/HotpotQA/加载脚本.py

# 只生成其中一个版本
py -3.12 data/HotpotQA/加载脚本.py --version full
py -3.12 data/HotpotQA/加载脚本.py --version small
```

默认输出到 `data/HotpotQA/`。使用 `--output-root PATH` 可以更换本地位置。

脚本重复执行时会核对 manifest、文件大小、记录数和 SHA-256。有效数据会直接复用；数据缺失、损坏或版本不一致时，脚本会拒绝覆盖。确认需要重建后使用：

```powershell
py -3.12 data/HotpotQA/加载脚本.py --version all --force
```

## 本地目录

```text
data/HotpotQA/
|-- README.md
|-- 加载脚本.py
|-- hotpotqa_data.py
|-- _manifest.py
|-- full/
|   |-- distractor/
|   |   |-- train.parquet
|   |   `-- validation.parquet
|   |-- fullwiki/
|   |   |-- train.parquet
|   |   |-- validation.parquet
|   |   `-- test.parquet
|   `-- manifest.json
`-- small/
    |-- train-5000.parquet
    `-- manifest.json
```

## 可复现规则

- 数据源：`hotpotqa/hotpot_qa`
- 固定 revision：`1908d6afbbead072334abe2965f91bd2709910ab`
- 小版本来源：`distractor/train`
- 小版本选择：按 `SHA-256("HotpotQA-small-v1:" + id)` 排序，取前 5,000 条
- 保存格式：Parquet，保留官方全部字段和嵌套结构

## 加载示例

```python
import pyarrow.parquet as pq

table = pq.read_table("data/HotpotQA/small/train-5000.parquet")
print(table.num_rows)  # 5000
print(table.column_names)
```

框架联调继续使用 `experiments/retrieval/adapters/hotpotqa.py` 中的 `adapt_hotpotqa`，不要在本目录重复实现字段转换。

## 提交检查

提交前确认 `full/`、`small/`、Hugging Face 缓存、模型权重、实验输出和大模型生成的过程文档没有进入暂存区：

```powershell
git status --short
git diff --cached --stat
git check-ignore data/HotpotQA/full data/HotpotQA/small
```
