# Retrieval Skill 原框架兼容性验证设计

## 目标

验证修改后的 BM25F 与 BGE V2 Retrieval Components 能继续被原框架发现、绑定、调用和执行。该验证只证明接口与运行链路兼容，不重复检索效果实验，也不评价生成质量。

## 验证链路

使用一条覆盖两个 Retrieval Components 的完整链路：

```text
RAGRequest
  -> Manage Skill
  -> Agentic RRFusion Skill
  -> Component 选择与编译
     -> BM25F Retriever
     -> BGE V2 Retriever
  -> RRFusion
  -> Grounded Generator
  -> RAGResult
```

Manage、Agentic RRFusion、RRFusion 算法和 Grounded Generator 均使用原框架实现。测试使用固定模型响应驱动原选择流程，使用确定性的假向量服务替代外部 BGE 服务，并使用现有小型文档样本，避免网络、模型下载和生成 API 费用。

## 验收标准

单个回归测试必须证明：

1. `run_rag()` 完整执行 Manage、Agentic 和 Component 选择阶段。
2. 选择结果绑定 `agentic-rrfusion`、BM25F、BGE V2 和原 Grounded Generator。
3. 两个 Retriever 接收兼容的 `RetrievalRequest` 并返回可融合的 `RetrievalResult`。
4. RRFusion 输出排序文档，并将这些文档传给原 Grounded Generator。
5. 最终结果包含非空 `answer`、排序后的 `documents`、选择记录、编译指令和 `retrieve/fuse -> generate` trace。
6. 测试不访问网络，不加载真实 BGE 权重，不依赖 API key。

## 范围外事项

- 不运行 HotpotQA、2WikiMultihopQA 或 TriviaQA 完整实验。
- 不重新选择参数或修改 Retrieval 默认参数。
- 不比较 Hit、Recall、AllSupport、MRR、EM 或 F1。
- 不修改 Manage、Agentic、Generator 或公共接口。
- 不将该兼容性测试解释为端到端答案质量提升证据。

