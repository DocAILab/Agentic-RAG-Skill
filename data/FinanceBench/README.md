# FinanceBench 数据准备

当前只接入 Hugging Face 上可公开获取的 FinanceBench 样本，并为以后补全量版本预留目录位。

公开数据固定到 Hugging Face revision
`e04404e3a97f69f79c14d42f24981a1c9c3bcd18`，避免上游 `main` 更新导致本地副本漂移。

## 版本

- `public/`：当前公开数据的完整副本
- `small/`：从 `public/` 中按稳定 SHA-256 顺序截取的确定性子集

## 目录

```text
data/FinanceBench/
|-- README.md
|-- 加载脚本.py
|-- financebench_data.py
|-- _manifest.py
|-- public/
|   |-- train.jsonl
|   `-- manifest.json
`-- small/
    |-- train.jsonl
    `-- manifest.json
```

## 生成

```powershell
python data/FinanceBench/加载脚本.py --version all
python data/FinanceBench/加载脚本.py --version small --small-limit 10
```

已存在版本的 revision 或抽样参数不一致时，脚本会拒绝复用。确认需要替换后显式增加
`--force`。

`small` 的抽样规则是：

- 以 `financebench_id` 为主键
- 计算 `SHA-256("FinanceBench-small-v1:" + id)`
- 按哈希排序后取前 `N` 条

## 预留

后续如果能拿到全量 FinanceBench，可以在同级目录继续增加新的版本目录，而不改当前 `public/` 和 `small/` 的输出契约。
