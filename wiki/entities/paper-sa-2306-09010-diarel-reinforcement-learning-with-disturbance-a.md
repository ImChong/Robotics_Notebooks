---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2306.09010"
venue: "arXiv 2023"
summary: "A reinforcement learning framework with disturbance awareness for robust sim-to-real policy transfer in robot control"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2306_09010_diarel-reinforcement-learning-with-distu.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# DiAReL

**DiAReL: Reinforcement Learning with Disturbance Awareness for Robust Sim2Real Policy Transfer in Robot Control** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 002/063** 篇，分组 **21 Reinforcement Learning-based Sim2Real**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A reinforcement learning framework with disturbance awareness for robust sim-to-real policy transfer in robot control

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- A reinforcement learning framework with disturbance awareness for robust sim-to-real policy transfer in robot control
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph PH1["阶段 1：训练逆动力学模型 Algorithm 1"]
        direction TB
        N1["名义 agent 在非随机化仿真中训练<br/>收集 o_k 与 a_k 序列"] --> N2["用掩码 M_u 与 M_y 构造数据对<br/>输入窗口 Y_k,L 到系统输入 u_k-L"]
        N2 --> N3["监督学习训练逆模型 MLP<br/>MSE 损失"]
    end
    subgraph PH2["阶段 2：干扰感知 on-policy RL Algorithm 2"]
        direction TB
        E1["受扰且随机延迟的随机化仿真<br/>观测 o'_k-h 延迟 h 在 0 到 H 之间"] --> E2["DOB：逆模型由输出窗口估计应有输入<br/>与实际施加输入作差得到 δ"]
        E2 --> E3["低通滤波，系数 β 由截止频率与步长得到<br/>得到估计干扰 d_hat"]
        E3 --> E4["拼接干扰增广观测<br/>延迟观测 + 最近 H+1 步估计干扰"]
        E4 --> E5["策略 π_θ 输出动作<br/>PPO 收集 T 步 rollout 后更新"]
        E5 --> E1
    end
    N3 --> E2
    E5 --> OUT["得到对随机化与干扰更鲁棒的策略<br/>用于 sim 到 real 迁移"]
```

图依据：论文 §III Methodology（Algorithm 1 逆模型训练、DAMDP 定义、Algorithm 2）/ Fig.1、Fig.3；PPO 设定见 §IV

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 002/063 |
| 分组 | 21 Reinforcement Learning-based Sim2Real |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2306.09010> |

## 核心机制（归纳）

### 策展导读要点

A reinforcement learning framework with disturbance awareness for robust sim-to-real policy transfer in robot control

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2306.09010) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **21 Reinforcement Learning-based Sim2Real** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DiAReL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **21 Reinforcement Learning-based Sim2Real**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_r2s2r_2306_09010_diarel-reinforcement-learning-with-distu.md`](../../sources/papers/sun_awesome_r2s2r_2306_09010_diarel-reinforcement-learning-with-distu.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2306.09010>
- [arXiv PDF](https://arxiv.org/pdf/2306.09010)：已通读方法部分（实验与附录以略读为准）：§I-II 背景、§III Methodology（含 Algorithm 1、2）、§IV 实验设置中的网络与算法设定；§V 实验结果仅略读

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2306.09010)
