# Group 数据格式

本目录只包含可直接加载的最终数据。下载、转换、抽样、测试、缓存和中间产物保存在仓库外的 `../../Preparation/`。运行框架不依赖该准备目录中的代码；只需当前仓库的数据文件。

## 当前数据

| 数据集 | Group 数 | 独立有效 QA | QA 总出现次数 | 每组 Task × QA | 每组 Corpus | 每组 Reference 范围 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 91 | 90,447 | 91,000 | 20 × 50 | 6,000 | 1,968–1,991 |
| 2WikiMultihopQA | 168 | 167,454 | 168,000 | 20 × 50 | 6,000 | 1,917–3,546 |
| TriviaQA RC | 102 | 76,220 | 102,000 | 20 × 50 | 6,000 | 973–6,000 |

总计 **361 个 Group、334,121 条独立有效 QA、361,000 次 QA 出现**，覆盖所固定版本训练源中的全部有效 QA。每个 Group 内 QA 唯一；不同 Group 允许重复采样。每组 distractor 数为 6,000 减去 reference 数，详细计数见 [catalog](groups/catalog.json)。

TriviaQA 完整训练源包含 **76,523 个独立 QuestionId**，合并 wiki/web 证据与答案变体后，保留 **76,220 条有效 QA**。另外 **303 条**因候选证据中没有有效答案别名匹配（`missing_weak_reference`）而排除。

FinanceBench 暂不纳入。以上规模是已发布 Group 的参数，不是框架硬限制。

原有 `*-train-001` 保持不变。扩建覆盖固定版本的完整训练源：先将已发布 QA 计入覆盖，再在每个新 Group 中放入尚未覆盖的 QA，按完整 reference 并集检查 6,000 篇容量，必要时复用低增量 QA 补足 1,000 题。重复仅发生在 Group 之间，不截断任何题的 gold，也不复制文档填充 Corpus。TriviaQA 在筛选有效 QA 前已合并全部训练分片中的 wiki/web 证据和答案变体；无有效 reference 的问题如实排除。Task 顺序和干扰文档选择由固定种子与 Group ID 决定。各组 `group.json` 记录源版本、选样策略、源序号、新增/复用数量和哈希。

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
  hotpotqa-train-002/...
  ...
  2wiki-train-001/...
  2wiki-train-002/...
  ...
  triviaqa-train-001/...
  triviaqa-train-002/...
  ...
```

各 Group 拥有独立的完整 Corpus 文件。内容允许跨 Group 重叠，但规范化内容集合不能完全相同；不同文件顺序、Group 名称或本地 ID 不构成语料差异。目录索引与组间重叠统计记录在 catalog 中。`coverage` 给出各数据集有效 QA 全集覆盖和重复出现次数；`corpus_overlap` 给出每对 Group 的语料重叠，`qa_overlap` 仅列出共享 QA 数大于零的组对。

## Corpus 记录

每条 JSONL 为一个原始段落或证据文档，本版本不额外切块：

```json
{"id":"doc-<sha256>","title":"...","text":"...","content_hash":"<sha256>","role":"reference","origin":"qa_context","source":{"dataset":"hotpotqa"}}
```

- `content_hash`：对 `[title.strip(), " ".join(text.split())]` 的紧凑 UTF-8 JSON（`ensure_ascii=false`）计算 SHA-256；`id` 为 `doc-` 加完整哈希。
- `role`：`reference` 表示至少被本组一个 QA 标为相关；`distractor` 表示未被本组 QA 标注为相关。某文档是 Group reference，不表示它对本组每个问题都相关。
- `origin`：`qa_context` 表示来自所选 QA 的原始文档；`supplement` 表示从其他问题的文档池补充。
- `source`：保留来源信息用于审计，运行时不会交给 Skill。

首先保留全部 reference，再按固定种子抽取非 reference 文档补足目标容量。原始候选全集没有必须保留的要求。容量小于完整 reference 并集、单组可用的不同 QA 不足 1,000 条、独立文档池不足均应拒绝构建。允许复用其他 Group 已使用的 QA；同一 Group 内不能重复 QA，同一 Corpus 内不能重复文档，也不能截断任何 gold。

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
