---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2109.12674"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2109_12674_metadrive-composing-diverse-driving-scen.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# MetaDrive

**MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 133/139** 篇，分组 **Transportation Environments**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

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
  ENV["MetaDrive 环境<br/>按用户配置创建 Managers"] --> MAP["Map Manager<br/>每个 episode 随机采样 PG 地图或真实地图"]
  subgraph PG["程序化生成 PG（Algorithm 1）"]
    BLK["随机选路块类型并随机化参数<br/>GetNewBlock"] --> SOCK["对齐 socket 并旋转<br/>与现有路网不相交则拼接"]
    SOCK --> BIG["Block Incremental Generation<br/>递归追加，失败则回退移除末块"]
  end
  REAL["导入真实数据<br/>Waymo / Argoverse 地图与交通流"] --> MAP
  BIG --> MAP
  MAP --> TRAF["Traffic Manager<br/>交通车辆 IDM 等规则策略"]
  TRAF --> OBJ["Object Manager<br/>随机散布障碍物"]
  OBJ --> AGT["Agent Manager<br/>被控车辆，由外部策略（如 RL）驱动"]
  AGT --> STEP["每步：各 Manager step()<br/>Bullet 物理仿真 + Panda3D 渲染"]
  STEP --> OBS["多模态观测 / 状态<br/>点云、RGB/深度、鸟瞰图、标量传感"]
  OBS --> POL["外部策略（RL / 人类控制）<br/>输出控制信号"]
  POL --> STEP
```

图依据：论文 §3 System Design（Fig. 2 Manager 组合、§3.2 场景组合工作流）、§4 Composing Diverse Driving Scenarios（四个基础 Manager 的顺序）、§4.1 与 Algorithm 1（PG 地图生成）、Fig. 1（多模态观测）；本篇为仿真平台/基准论文，图仅表示平台的场景组合流程。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 133/139 |
| 分组 | Transportation Environments |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2109.12674> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Transportation Environments。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2109.12674) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Transportation Environments** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MetaDrive」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/lc_awesome_sim2real_2109_12674_metadrive-composing-diverse-driving-scen.md`](../../sources/papers/lc_awesome_sim2real_2109_12674_metadrive-composing-diverse-driving-scen.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/2109.12674>
- [arXiv PDF](https://arxiv.org/pdf/2109.12674)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§2 相关工作、§3 系统设计、§4.1 地图生成（Algorithm 1）；§5 基准实验以略读为准

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/2109.12674)
