# Framework 文档索引

本文件是 Framework 文档入口。项目介绍、开发方法和执行逻辑分别维护在以下文档中。

| 阅读目的 | 文档 |
|---|---|
| 理解项目、安装配置、校验数据、运行 Group | [English README](../README.md) · [中文 README](../README.zh-CN.md) |
| 维护模块、配置与公开 API，扩展 Skill，新增数据，测试与排障 | [开发文档 Word](../docs/DEVELOPMENT.docx) |
| 了解执行时序、全部 Skill、优化接受/回滚、缓存、评估与报告 | [技术文档 Word](../docs/TECHNICAL_DESIGN.docx) |
| 查看发布数据的存储格式、来源和标签口径 | [数据格式](../data/README.md) |

## 实现入口

- Skill 包发现、元数据与能力约束：[spec.py](spec.py)。
- 请求与组件上下文协议：[interfaces.py](interfaces.py)。
- Manage 指导和分层选择：[selection.py](selection.py)。
- 绑定、编译、执行与共享 Corpus 上下文：[compiler.py](compiler.py)。
- Group 数据加载与校验：[data_groups.py](data_groups.py)。
- 检索与生成评估：[evaluation/](evaluation)。
- Group/Task 自优化入口：[task_flow.py](../self_optimization/task_flow.py)。
- Policy 提案与 Skill 副本修订：[optimizer.py](../self_optimization/optimizer.py)、[workspace.py](../self_optimization/workspace.py)。

`SKILL.md` 提供模型阅读的使用指导，`ragskill.yaml` 提供机器可校验的包契约，`scripts/` 实现执行逻辑。各 Skill 的输入输出、参数、依赖和流程见[完整 Skill 说明](../docs/TECHNICAL_DESIGN.docx)。
