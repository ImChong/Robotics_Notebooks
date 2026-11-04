---
type: entity
tags: [paper, manipulation, vla, benchmark]
status: draft
updated: 2026-11-04
project_id: vlalagbench
arxiv: "2611.03103"
related:
  - ./paper-vla-hwbench-edge-hardware-latency.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/vlalagbench_arxiv_2611_03103.md
summary: "VLALagBench：评估推理延迟、动作执行模式与动态操作成功率的关系。"
---

# VLALagBench：VLA 推理延迟何时及为何重要

> 草稿：核心评测方法已核对；全文及附录尚未通读完成，数值结论和适用边界待复核。

## 一句话定义

VLALagBench 通过控制动态目标速度、推理延迟和动作执行方式，研究 VLA 操作成功率对延迟的敏感性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| LIBERO | Lifelong Benchmark for Robot Learning | 主要仿真任务基础 |
| RMS | Root Mean Square | 均方根，用于评估预测误差 |

## 流程总览

```mermaid
flowchart TB
  config["任务 / 目标速度 / VLA 策略"] --> env["LIBERO 动态目标环境"]
  env --> observe["获取观测并注入推理延迟"]
  observe --> mode["同步 / lazy 异步 / continuous 异步执行"]
  mode --> result["记录任务成功率"]
  result --> model["校准经验模型，预测配置成功率"]
```

## 执行模式与评测

同步模式等待推理时保持当前位置；lazy 异步在旧动作执行期间请求下一块并跳过推理期间过时的动作；continuous 异步在推理完成后尽快发起下一次请求。异步可减少停顿，但更频繁推理与动作块切换也有成本。主实验使用 20 Hz 控制、0–250 ms 注入延迟，每个成功率点 200 次 rollout；覆盖释放、放置、抓取和关节物体操作。

## 结论

延迟影响必须结合任务、执行模式和目标运动评估。

- 将观测陈旧与等待期间机器人停顿分别考虑。
- 异步执行可能改善成功率，但收益随模型与任务变化。
- 经验模型需按任务与策略校准，不能当作跨任务保证。

## 源码运行时序图

不适用：论文给出的代码链接截至 2026-11-04 返回 404，作者声明代码将发布。

## 关联页面

- [VLA-HWBench](./paper-vla-hwbench-edge-hardware-latency.md)
- [视觉语言动作模型](../methods/vla.md)

## 参考来源

- [论文归档及 v1 原文](../../sources/papers/vlalagbench_arxiv_2611_03103.md)
