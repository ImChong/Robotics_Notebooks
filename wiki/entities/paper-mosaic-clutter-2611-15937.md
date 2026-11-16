---
type: entity
project_id: mosaic-clutter-2611-15937
tags: [paper, vla, spatial-reasoning, manipulation]
status: complete
updated: 2026-11-16
arxiv: "2611.15937"
project: https://mmahdavian.github.io/MOSAIC/
original:
  - ../../sources/papers/mosaic-clutter-2611-15937.md
sources:
  - ../../sources/papers/mosaic-clutter-2611-15937.md
  - ../../sources/sites/mosaic-clutter-2611-15937.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "将目标、放置位置和周围障碍物区分建模，用几何 token 与障碍感知注意力改进操作动作。"
---

# MOSAIC: Modular Obstacle and Spatial-Aware Framework for Manipulation In Cluttered Scenes｜MOSAIC：杂乱场景目标、落点与避障空间建模

## 一句话定义

**MOSAIC：杂乱场景目标、落点与避障空间建模**：将目标、放置位置和周围障碍物区分建模，用几何 token 与障碍感知注意力改进操作动作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| MOSAIC | Modular Obstacle and Spatial-Aware Framework | 目标和避障几何融合架构 |
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| RGB-D | Red Green Blue and Depth | 彩色深度观测 |

## 为什么重要

将目标、放置位置和周围障碍物区分建模，用几何 token 与障碍感知注意力改进操作动作。 需要结合任务边界、数据/算力条件及作者真实实现来判断，而不是只看展示视频。

## 核心信息

| 证据项 | 官方信息 |
|---|---|
| 论文 | [arXiv:2611.15937v1](https://arxiv.org/abs/2611.15937) |
| 项目页 | [原始页面](https://mmahdavian.github.io/MOSAIC/) |
| 源码 | 无经核实的官方可运行完整发布 |

## 核心原理

VLA 骨干输出目标/落点语义 mask，冻结的三维解码器提供深度和置信度；针对目标及落点生成各 17 个几何 token，后层动作专家接收这些几何条件；障碍物使用专门的邻近注意力引导。

### 流程总览

```mermaid
flowchart LR
  V0["多视角视觉与指令"]
  V1["目标/目的地/障碍物 mask"]
  V2["深度及 4x4 几何网格"]
  V3["动作专家注意力引导"]
  V4["杂乱场景避碰操作"]
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

主要定量结果来自 RoboPRO 杂乱操作仿真，真机部分以定性展示为主；不能当作大样本真机成功率提升。

## 结论

**将目标、放置位置和周围障碍物区分建模，用几何 token 与障碍感知注意力改进操作动作。**

1. 先复核指标口径与论文约束，再比较其他策略或系统。
2. 将作者验证的模块和尚未发布/未复现的模块分开记录。
3. 把误差、失败模式与传感器/真实场景分布偏移作为后续验收依据。

## 局限与风险

对深度置信度、语义 mask 和障碍物检测敏感，真实几何误差与动作控制延迟仍需验证；项目网页抓取未成功。

## 关联页面

- [29 篇具身论文研究地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder 操作任务难度分层](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [论文原文归档](../../sources/papers/mosaic-clutter-2611-15937.md)
- [官方项目及开源状态](../../sources/sites/mosaic-clutter-2611-15937.md)

## 推荐继续阅读

- [原始论文 HTML](https://arxiv.org/html/2611.15937v1)
- [原文 PDF](https://arxiv.org/pdf/2611.15937)
