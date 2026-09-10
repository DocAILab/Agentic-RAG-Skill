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

## 多 Task 切分

当需要把一个大分片切成多个互斥、可顺序自优化的小 Task 时，使用
`self_optimization/split_task_data.py`。它对全量有效样本按
`sha256("{dataset}:{sample_id}")` 确定性排序（与 `experiments/retrieval`
的抽样 manifest 规则一致），再按 K 切成互不重叠的 task 文件；每个 task
和整档 suite 都带可校验 manifest。

```powershell
python -B -m self_optimization.split_task_data `
  --dataset triviaqa --split validation --task-size 200 20
```

默认输出到 `data/task_flow/triviaqa-validation/`（本地生成，git 忽略）：

```text
triviaqa-validation/
|-- universe.manifest.json        # 全量有效样本清单（哈希序）
|-- records.jsonl                 # 规范化全量记录缓存
|-- k200/task-*.jsonl ...         # 200 题一档，88 个 Task
`-- k20/task-*.jsonl ...          # 20 题一档，873 个 Task
```

指定 `--task-size 200` 只生成一档；加 `--write-task-flow` 会为每档写出
`self_optimization/task_flow.triviaqa-k*.yaml`（git 忽略），可直接复制为
本地 `task_flow.yaml` 使用。默认只收录带答案与弱标签的有效样本；
rc validation 共 17,944 行（wiki/web 证据变体，唯一 QuestionId 约 9,960），
默认过滤剔除 112 行无弱标签样本与 380 行完全重复记录后得到 17,452 个有效
样本行；`--no-require-labels` 可放宽该过滤。
