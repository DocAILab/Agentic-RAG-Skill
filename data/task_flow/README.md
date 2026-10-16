# Task Flow Data

该目录保存多任务 Skill 自优化流程使用的小型、规范化本地数据。默认跨数据集任务流包含 HotpotQA、2WikiMultihopQA 和 TriviaQA validation 各一个样本，以及 FinanceBench train 一个样本；2Wiki 专用任务流还包含按官方推理类型固定切分的四个 task，每个 task 10 个样本。每条 JSONL 记录包含：

- `id`、`dataset`、`question`
- `answers`
- `documents`
- `relevant_document_ids`
- `label_type`
- 2Wiki task 额外包含 `question_type`

对应的 `*.manifest.json` 记录源数据集、split、原始行号、样本 ID 和按 LF 规范化后的 UTF-8 内容 SHA-256。可使用现有 adapter 重新构建，例如：

```powershell
python -B -m self_optimization.build_task_data --dataset hotpotqa --split validation --examples 1 --output data/task_flow/hotpotqa-validation.jsonl
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 1 --output data/task_flow/2wiki-validation.jsonl
python -B -m self_optimization.build_task_data --dataset triviaqa --split validation --examples 1 --output data/task_flow/triviaqa-validation.jsonl
python -B -m self_optimization.build_task_data --dataset financebench --split train --examples 1 --output data/task_flow/financebench-train.jsonl
```

## 2Wiki reasoning-type tasks

2WikiMultihopQA validation（官方 `dev`）样本按四种推理类型拆分：

- `compositional`
- `inference`
- `comparison`
- `bridge_comparison`（同时接受源数据或命令行中的 `bridge-comparison`）

每个 task 从 `data/2WikiMultihopQA/demo/sample_manifest.json` 固定的 100 个样本 ID 中按 manifest 顺序选择前 10 个对应类型样本。四个 task 因题型互斥而没有重复 ID；task manifest 同时记录固定样本 manifest 的摘要和输出 SHA-256。

重新构建示例：

```powershell
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 10 --question-type compositional --sample-manifest data/2WikiMultihopQA/demo/sample_manifest.json --output data/task_flow/2wiki-compositional-validation.jsonl
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 10 --question-type inference --sample-manifest data/2WikiMultihopQA/demo/sample_manifest.json --output data/task_flow/2wiki-inference-validation.jsonl
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 10 --question-type comparison --sample-manifest data/2WikiMultihopQA/demo/sample_manifest.json --output data/task_flow/2wiki-comparison-validation.jsonl
python -B -m self_optimization.build_task_data --dataset 2wiki --split validation --examples 10 --question-type bridge_comparison --sample-manifest data/2WikiMultihopQA/demo/sample_manifest.json --output data/task_flow/2wiki-bridge-comparison-validation.jsonl
```

准备本地 `self_optimization/settings.yaml` 后，可先进行每个 task 一条样本的冒烟运行：

```powershell
python -B run_task_flow.py --config self_optimization/task_flow.2wiki.example.yaml --examples-per-task 1 --run-id 2wiki-multitask-smoke
```

真实模型运行结果保存在被 Git 忽略的 `self_optimization/runs/`，不得提交 API Key、`settings.yaml` 或运行副本。

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

FinanceBench 样本来自固定 revision 的公开 `train` split，仅包含标注的证据页，不代表完整 PDF 检索语料。
