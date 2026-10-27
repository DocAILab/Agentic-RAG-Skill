# Group 数据格式

本目录只包含可直接加载的最终数据。下载、转换、抽样、测试、缓存和中间产物保存在仓库外的 `../Preparation/`。运行框架不依赖该准备目录中的代码；只需当前仓库的数据文件。

## 当前数据

| Group | 来源 | Task | 每 Task QA | Corpus | Reference | Distractor |
|---|---|---:|---:|---:|---:|---:|
| `hotpotqa-train-001` | HotpotQA train | 20 | 50 | 6,000 | 1,977 | 4,023 |
| `2wiki-train-001` | 2WikiMultihopQA train | 20 | 50 | 6,000 | 2,311 | 3,689 |
| `triviaqa-train-001` | TriviaQA RC train | 20 | 50 | 6,000 | 5,507 | 493 |

FinanceBench 暂不纳入。以上规模是已发布 Group 的参数，不是框架硬限制。

每组取固定源版本、源顺序中前 1,000 个有效独立 QA。TriviaQA 已跨训练分片合并选中问题的 wiki/web 证据变体。随后按固定种子确定 Task 分配，QA 在组内互斥。来源、版本、过滤口径、文件哈希和实际计数记录在各组 `group.json`。

## 文件布局

```text
data/groups/
  catalog.json
  hotpotqa-train-001/
    group.json
    corpus/part-0000.jsonl.gz
    corpus/part-0001.jsonl.gz
    corpus/part-0002.jsonl.gz
    tasks/task-0001.jsonl
    ...
    tasks/task-0020.jsonl
  2wiki-train-001/...
  triviaqa-train-001/...
```

各 Group 拥有独立的完整 Corpus 文件。内容允许跨 Group 重叠，但规范化内容集合不能完全相同；不同文件顺序、Group 名称或本地 ID 不构成语料差异。目录索引与组间重叠统计记录在 catalog 中。

## Corpus 记录

每条 JSONL 为一个原始段落或证据文档，本版本不额外切块：

```json
{"id":"doc-<sha256>","title":"...","text":"...","content_hash":"<sha256>","role":"reference","origin":"qa_context","source":{"dataset":"hotpotqa"}}
```

- `content_hash`：对 `[title.strip(), " ".join(text.split())]` 的紧凑 UTF-8 JSON（`ensure_ascii=false`）计算 SHA-256；`id` 为 `doc-` 加完整哈希。
- `role`：`reference` 表示至少被本组一个 QA 标为相关；`distractor` 表示未被本组 QA 标注为相关。某文档是 Group reference，不表示它对本组每个问题都相关。
- `origin`：`qa_context` 表示来自所选 QA 的原始文档；`supplement` 表示从其他问题的文档池补充。
- `source`：保留来源信息用于审计，运行时不会交给 Skill。

首先保留全部 reference，再按固定种子抽取非 reference 文档补足目标容量。原始候选全集没有必须保留的要求。容量小于 reference 数、有效 QA 不足、补充池不足均应拒绝构建，不能截断 gold 或重复填充。

## QA 记录

```json
{"id":"hotpotqa:<source-id>","question":"...","answers":["..."],"relevant_document_ids":["doc-<sha256>"],"label_type":"supporting_facts","metadata":{}}
```

QA 不内嵌文档正文或候选子集，所有 QA 检索其 Group 的完整 Corpus。`answers` 与 `relevant_document_ids` 用于评估；答案别名仍保存在 `answers` 中。`metadata` 可记录源 ID、题型和来源信息。

HotpotQA、2Wiki 使用 `supporting_facts` 标签；TriviaQA 使用 `weak_answer_alias`。TriviaQA 标签来自该问题原始证据中的答案别名匹配，不是全 Corpus 的穷尽相关性标注。补充背景文档未被标为 reference，不代表已人工证实语义不相关；不同标签口径的评估应分别报告。

## Manifest 与加载校验

`group.json` 使用 `schema_version: 1`，包含 `group_id`、`dataset`、`source`、`corpus`、`tasks`、`task_count`、`qa_count`、`qa_per_task` 和 `build`。

`corpus.files` 列出相对文件路径、记录数及原始文件字节 SHA-256；`document_count`、`reference_count`、`distractor_count`、`target_size` 描述实际规模。`corpus.fingerprint` 对排序后的 `content_hash` 集合逐行连接（包括末尾 LF）计算 SHA-256，与文档顺序和 Group 名称无关。`tasks` 明确任务顺序、文件路径、QA 数量与文件 SHA-256。

加载器校验路径不能逃逸 Group 目录，检查文件哈希、记录数、QA/文档 ID 唯一性、证据覆盖、内容指纹，以及 reference 标记与全部 QA gold 并集的一致性。文本文件固定 UTF-8/LF，压缩文件固定原始字节哈希。

## 执行视图

`framework.data_groups.load_group()` 校验并加载一次 Corpus，使用 `shuffle_seed` 打乱，并生成只读文档视图。该视图仅包含 `id/title/text`：角色、来源、关联 QA 等处理标记不会进入检索或生成请求；答案及证据 ID 留在评估侧。

Group 内所有 Task 使用同一个语料成员集合。固定种子的打乱可复现，不改变 Group 身份。每个 Task 执行其全部 QA 后优化一次，后续 Task 继承工作副本中的 Skill 修订。
