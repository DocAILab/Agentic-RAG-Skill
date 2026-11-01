# Task 内自适应并行

一个 Task 的 QA 共用一次规划结果和同一版 Skill，题目之间没有答案依赖，可以同时执行。Task 之间保持顺序：

```text
当前 Skill → Task 统一规划 → 并行执行全部 QA → 汇总评估
          → 一轮 Skill 优化和运行验证 → 下一 Task 重新规划
```

优化开始前会等待本 Task 的全部主 QA 完成。结果、评估和 Policy 反馈按原始题目顺序保存，开始/完成事件记录真实执行顺序。每题使用独立的组件调用状态、证据元数据和日志上下文；共享只读 Corpus、同步缓存和模型资源。

## 配置和运行

在 `self_optimization/task_flow.yaml` 的 `workflow` 下设置：

```yaml
workflow:
  qa_execution:
    mode: adaptive
    min_workers: 1
    initial_workers: 2
    max_workers: 8
    adjustment_window: 4
    throughput_tolerance: 0.05
    latency_factor: 2.0
```

| 字段 | 含义 |
|---|---|
| `mode` | `serial` 每次执行一题；`fixed` 使用固定并发上限；`adaptive` 根据完成窗口调整并发上限。 |
| `min_workers` | 自适应并发下限。 |
| `initial_workers` | 自适应起始并发。 |
| `max_workers` | 同时执行的 QA 上限，不是单次模型请求数或 GPU batch size。 |
| `adjustment_window` | 调整前收集的最少完成题数。 |
| `throughput_tolerance` | 判断吞吐量变化时使用的相对容差。 |
| `latency_factor` | 判断题目延迟膨胀时使用的倍数阈值。 |

旧配置省略 `qa_execution` 时保持串行；提供的示例启用自适应模式。可从命令行覆盖模式和上限：

```sh
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --qa-mode adaptive --qa-max-workers 8
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --qa-mode fixed --qa-max-workers 4
python -B -m self_optimization --group data/groups/hotpotqa-train-001 --qa-mode serial
```

修改配置或并行模式后使用新的 run ID；恢复运行需使用原配置与源码版本，避免将不同调度实验混入同一份证据。

## 自适应调度的目标

调度器使用实测完成吞吐量、QA 延迟和调用重试压力，逐步探测可用并发；压力上升时降低后续提交数量。降低上限不会强行终止已经执行的题目。

这是有上限的吞吐量调节，不能保证硬件利用率最大。API 的远端 GPU 利用率不可见，问题难度也会影响窗口统计。`max_workers` 需要结合服务额度、内存和实测吞吐量设置。建议先比较并发 2、4、8 的同一组题，而非直接把上限设为题目总数。

本地 BGE 重排继续使用共享模型和推理锁，避免同时加载多个模型或叠加推理显存峰值。QA 并行可以让 API 等待与本地重排重叠；跨题 GPU 批处理是另一项优化，本模式不把它当作已经实现的收益。

## 故障和扩展边界

- 任一主 QA 最终执行失败，会停止提交新题，取消尚未开始的工作并等待已开始的工作结束。该 Task 不进行优化，也不开始下一 Task；失败证据保留。恢复时仍按 Task 事务重新执行，不把未完成题当作成功。
- 调度器不会提前返回一个仍有后台线程运行的失败 Task。运行中的阻塞 HTTP 请求不能被 Python 线程强行取消，失败收尾最长仍受模型客户端的超时和重试预算约束。
- 内置模型客户端支持并发调用统计。注入自定义 `ModelClient`、transport 或封装器时，调用方必须保证它们支持并发，或选择串行模式；每次请求的统计应当独立，不能用共享累计计数的前后差冒充单次统计。
- 历史 `Preparation` 实验启动脚本中的外部计量封装器有共享状态，不能直接用来评估并行。新实验使用这里的原生入口及原生采集；旧运行和证据不改写。
- Task 内并行不改变 RAG Skill 的控制流、模型选择、Critic、评估、优化次数，以及代码执行、接口和修改边界检查。

Task 报告及事件保留调度配置、并发峰值、吞吐量窗口和调整原因，用于比较实际收益。离线并发回归验证执行契约；模型服务性能和最大可用并发仍需真实实验测量。
