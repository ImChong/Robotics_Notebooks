---
type: entity
project_id: so101-toilet-cleaning-2611-15381
tags: [paper, mobile-manipulation, vla, sim2real]
status: complete
updated: 2026-11-16
arxiv: "2611.15381"
project: https://so101-turtlebot4.github.io/mobile-bimanual-cleaning/
original:
  - ../../sources/papers/so101-toilet-cleaning-2611-15381.md
sources:
  - ../../sources/papers/so101-toilet-cleaning-2611-15381.md
  - ../../sources/sites/so101-toilet-cleaning-2611-15381.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "在 TurtleBot 4 与双 SO-101 平台，用统一演示数据对比多种 VLA 完整清洁长程任务。"
---

# Low-Cost Mobile Bimanual Manipulation for Long-Horizon Toilet Cleaning｜低成本移动双臂马桶清洁长程操作

## 一句话定义

**低成本移动双臂马桶清洁长程操作**：在 TurtleBot 4 与双 SO-101 平台，用统一演示数据对比多种 VLA 完整清洁长程任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SO101 | SO-101 | 开源低成本机械臂 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| ROS2 | Robot Operating System 2 | ROS 2 |

## 为什么重要

在 TurtleBot 4 与双 SO-101 平台，用统一演示数据对比多种 VLA 完整清洁长程任务。 需要结合任务边界、数据/算力条件及作者真实实现来判断，而不是只看展示视频。

## 核心信息

| 证据项 | 官方信息 |
|---|---|
| 论文 | [arXiv:2611.15381v1](https://arxiv.org/abs/2611.15381) |
| 项目页 | [原始页面](https://so101-turtlebot4.github.io/mobile-bimanual-cleaning/) |
| 源码 | 无经核实的官方可运行完整发布 |

## 核心原理

双 SO-101 与 TurtleBot4 通过 Zenoh–ROS 2 桥和 LeRobot 采集 250 段真机演示；对 π0、π0.5、SmolVLA、GR00T 分别微调，另外测试进度信号 SARM 和短记忆 MEM 对阶段切换的作用。

### 流程总览

```mermaid
flowchart LR
  V0["双臂移动机器人硬件"]
  V1["250 段真机示教"]
  V2["统一 VLA 微调"]
  V3["分阶段清洁执行"]
  V4["严格/宽松/人工协助评测"]
  V0 --> V1
  V1 --> V2
  V2 --> V3
  V3 --> V4
```

该图为研究问题与方法/控制流，实际软件调用须看源码时序图及开放边界。

## 源码运行时序图

**不适用：** 官方源码尚无经过核实的可运行完整入口，或当前公开的代码不具备端到端可执行部署能力，故不虚构 sequenceDiagram。

## 工程实践

1. 检查输入观察、动作/轨迹表示、数据来源和实验任务的部署条件是否一致。
2. 等待作者公开或确认源码后再写运行时序。
3. 将仿真指标、真机成功率、推理延迟与安全/碰撞检查分开汇报。

## 实验与评测

π0 在该任务下严格自主成功 60%、宽松完成 70%、有人短暂干预时 90%；进度/记忆增强没有稳定优于对应基线。

## 结论

**在 TurtleBot 4 与双 SO-101 平台，用统一演示数据对比多种 VLA 完整清洁长程任务。**

1. 先复核指标口径与论文约束，再比较其他策略或系统。
2. 将作者验证的模块和尚未发布/未复现的模块分开记录。
3. 把误差、失败模式与传感器/真实场景分布偏移作为后续验收依据。

## 局限与风险

结果集中于同一个硬件、房间和五阶段马桶清洁任务；严格成功与人工介入成功分母不同，代码和数据在项目页仍标 Coming Soon。

## 关联页面

- [29 篇具身论文研究地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder 操作任务难度分层](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [论文原文归档](../../sources/papers/so101-toilet-cleaning-2611-15381.md)
- [官方项目及开源状态](../../sources/sites/so101-toilet-cleaning-2611-15381.md)

## 推荐继续阅读

- [原始论文 HTML](https://arxiv.org/html/2611.15381v1)
- [原文 PDF](https://arxiv.org/pdf/2611.15381)
