---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2010.09776"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2010_09776_smarts-scalable-multi-agent-reinforcemen.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# SMARTS

**SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 135/139** 篇，分组 **Transportation Environments**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  DSL["场景规范（Python DSL）：地图、路线、车辆、flows、bubble"] --> BT["背景交通提供者（SUMO）：全图提供背景车流"]
  DSL --> BUB["Bubble：时空条件区域，A Zone / 过渡区 / B Zone"]
  BT -->|"车辆触及 bubble 外膜"| BUB
  BUB -->|"移交控制，实例化车辆与传感器模型、启动智能体进程"| SA["社会智能体（来自 Social Agent Zoo）"]
  EGO["学习中的 ego 智能体（Ray / RLlib 等 MARL 库训练）"] --> VP["车辆物理提供者（Bullet）：油门刹车转向 / 轨迹跟踪 / 车道跟随等控制接口"]
  SA --> VP
  MP["运动规划提供者（SUMO 运动模型 / Bezier / OpEn）"] --> SA
  VP --> STEP["仿真步进由各 provider 共同编排"]
  BT --> STEP
  STEP --> OBS["观测：动态物体列表、BEV 占据栅格、RGB、自车状态、道路结构；奖励：沿路线行驶距离"]
  OBS --> EGO
```

图依据：论文 §3（Fig.3 SMARTS 架构、Fig.4 Bubble、Distributed Computing）与 §4（Observation / Action / Reward、Algorithms & Baselines）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 135/139 |
| 分组 | Transportation Environments |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2010.09776> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2010.09776) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Transportation Environments** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SMARTS」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Transportation Environments**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_2010_09776_smarts-scalable-multi-agent-reinforcemen.md`](../../sources/papers/lc_awesome_sim2real_2010_09776_smarts-scalable-multi-agent-reinforcemen.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/2010.09776>
- [arXiv PDF](https://arxiv.org/pdf/2010.09776)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 设计目标、§3 架构与实现、§4 对 MARL 的支持；§5 实验略读。

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/2010.09776)
