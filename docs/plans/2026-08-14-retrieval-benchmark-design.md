# BM25 与 BGE-Large 检索评测设计

## 目标与边界

第一阶段完善现有 BM25 与 Vector Retriever，同时建立统一、可复现的三数据集检索评测流水线。现有 `run(inputs, context)`、`RetrievalRequest` 和 `RetrievalResult` 接口保持兼容；每个问题只在数据集为该问题提供的候选文档中检索，不建设全局索引，也不引入 Elasticsearch、FAISS 或生成模型调用。

## 组件设计

BM25 对英文文本执行 Unicode NFKC 规范化、大小写折叠和稳定分词，同时检索 `title` 与 `text`，默认提高标题词频权重。`k1`、`b`、`title_boost` 和 `top_k` 可配置；空输入、缺失文本和相同分数均产生确定性结果。

Vector Retriever 默认使用 `BAAI/bge-large-en-v1.5`。查询添加 BGE 检索指令，文档不添加；向量在组件内校验并归一化。文档可携带预计算 `embedding`，仅对缺失向量的文档调用 `context.embed()`。数量、维度、非有限值和零向量错误必须给出明确提示。

## 统一数据结构

三个适配器流式输出 `RetrievalExample`：

```text
RetrievalExample
├── id
├── query
├── documents[]
│   ├── id
│   ├── title
│   └── text
├── relevant_document_ids[]
└── label_type
```

- HotpotQA：`context` 生成候选文档，`supporting_facts.title` 生成强相关标签。
- 2WikiMultihopQA：`context` 与 `supporting_facts` 生成候选文档和强相关标签。
- TriviaQA：`entity_pages` 与 `search_results` 生成候选文档；答案别名出现在证据文档中时生成弱相关标签，汇总结果单独标记为 `weak_answer_alias`。
- 无公开 gold 的测试样本保留检索结果，但指标为 `null`，不得构造伪标签。

## 指标与输出

每个有标签样本计算 `Hit@1/5/10`、`Recall@1/5/10`、`AllSupport@1/5/10` 和 `MRR`。逐条结果追加到 JSONL；checkpoint 原子写入运行参数、已处理数量和最近样本；恢复时读取已有结果 ID 并跳过，避免重复。最终 summary JSON 记录有效、无标签、无效和失败样本数，并按 `label_type` 分组，避免将强标签与弱标签混合解释。

## 验证策略

CI 使用固定小样本和假向量服务覆盖组件、适配器、指标、断点恢复和 CLI 参数。真实 BGE 仅作为可选的小样本集成测试；完整数据集评测不进入 CI，也不调用 DeepSeek。
