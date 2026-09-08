# Task Flow Data

该目录保存多任务 Skill 自优化流程使用的小型、规范化本地数据。当前包含 HotpotQA、2WikiMultihopQA 和 TriviaQA validation 各一个样本。每条 JSONL 记录包含：

- `id`、`dataset`、`question`
- `answers`
- `documents`
- `relevant_document_ids`
- `label_type`

对应的 `*.manifest.json` 记录源数据集、split、原始行号、样本 ID 和按 LF 规范化后的 UTF-8 内容 SHA-256。可使用现有 adapter 重新构建，例如：

```powershell
python -B -m self_optimization.build_task_data --dataset hotpotqa --split validation --examples 1 --output data/task_flow/hotpotqa-validation.jsonl
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 1 --output data/task_flow/2wiki-validation.jsonl
python -B -m self_optimization.build_task_data --dataset triviaqa --split validation --examples 1 --output data/task_flow/triviaqa-validation.jsonl
```
