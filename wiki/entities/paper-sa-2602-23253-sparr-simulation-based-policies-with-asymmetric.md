---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2602.23253"
venue: "arXiv 2026"
summary: "Simulation-based policies with asymmetric real-world residuals for assembly tasks"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2602_23253_sparr-simulation-based-policies-with-asy.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# SPARR

**SPARR: Simulation-based Policies with Asymmetric Real-world Residuals for Assembly** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 017/063** 篇，分组 **23 Domain Randomization & Adaptation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Simulation-based policies with asymmetric real-world residuals for assembly tasks

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Simulation-based policies with asymmetric real-world residuals for assembly tasks
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S["仿真：低维状态观测（关节角、末端位姿、目标位姿及其差）+ 稠密模仿奖励"] --> PB["PPO 预训练基策略 π_b（Isaac Lab）"]
    PB --> ZS["真实世界零样本部署：Grounding DINO + SAM2 + FoundationPose 估计插座位姿作目标"]
    ZS --> DC["采集演示：π_b 均值动作 + 从 N(0, σ_t) 采样的残差，保留成功轨迹"]
    DC --> DB["Demo Buffer"]
    RL["RL Buffer（在线交互）"] --> UP["RLPD：每次更新从两个 buffer 等量采样"]
    DB --> UP
    UP --> PR["残差策略 π_r：输入真实观测（本体量、力/力矩、两路腕部 RGB）+ 基动作 a_b；稀疏成功奖励"]
    PR --> CA["执行 a = a_b + a_r"]
    CA --> RL
    CA --> HQ["用时短于演示中位数的成功轨迹加入 Demo Buffer"]
    HQ --> DB
```

图依据：论文 §III-B–III-E / Fig.1、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 017/063 |
| 分组 | 23 Domain Randomization & Adaptation |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.23253> |

## 核心机制（归纳）

### 策展导读要点

Simulation-based policies with asymmetric real-world residuals for assembly tasks

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.23253) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **23 Domain Randomization & Adaptation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SPARR」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **23 Domain Randomization & Adaptation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/sun_awesome_r2s2r_2602_23253_sparr-simulation-based-policies-with-asy.md`](../../sources/papers/sun_awesome_r2s2r_2602_23253_sparr-simulation-based-policies-with-asy.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2602.23253>
- [arXiv PDF](https://arxiv.org/pdf/2602.23253)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III 方法（B 仿真预训练、C 真实部署与策略形式、D 演示采集、E 残差策略学习）、§IV-A/B 实验设定

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2602.23253)
