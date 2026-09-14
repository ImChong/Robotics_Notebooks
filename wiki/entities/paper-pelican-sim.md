---
type: entity
tags:
  - paper
  - world-model
  - simulation
  - x-humanoid
status: complete
updated: 2026-09-14
arxiv: "2609.12036"
code: https://github.com/Open-X-Humanoid/Pelican-Sim1.0
related:
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
  - ./paper-dynin-robotics.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/pelican-sim_arxiv_2609_12036.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "28 维统一动作空间 + URDF 渲染动作视频 + 稀疏 MoE + 四步 rollout；约百万轨迹上支持数据生成、策略评测、动作选择与策略改进。"
---

# Pelican-Sim 1.0（arXiv:2609.12036）

**Pelican-Sim 1.0**（[Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence](https://arxiv.org/abs/2609.12036)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。世界模型价值在于能否被策略训练与决策使用；Pelican-Sim 把 WM 变成数据生成、评测与选动作工具。

## 一句话定义

**28 维统一动作空间 + URDF 渲染动作视频 + 稀疏 MoE + 四步 rollout。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 世界模型价值在于能否被策略训练与决策使用；Pelican-Sim 把 WM 变成数据生成、评测与选动作工具。
- 开源状态：**待核实**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12036](https://arxiv.org/abs/2609.12036) |
| **项目页** | https://zoushilong1024.github.io/Pelican-Sim1.0/ |
| **代码** | https://github.com/Open-X-Humanoid/Pelican-Sim1.0 |
| **开源** | **待核实** |
| **文内指标** | RoboTwin 上 50 demo + 500 生成轨迹成功率 70%→93%；策略评测 Pearson 0.994；动作选择相对增益 47.7%。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | RoboTwin 上 50 demo + 500 生成轨迹成功率 70%→93%；策略评测 Pearson 0.994；动作选择相对增益 47.7%。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **物理引擎仿真器（[MuJoCo](./mujoco.md) / [Isaac Lab](./isaac-lab.md) 一系）** — 靠显式刚体动力学与接触求解推进状态，保真度由建模精度决定；Pelican-Sim 用 **学出来的世界模型** 推进，靠约百万轨迹的数据覆盖换泛化。取舍面见 [MuJoCo vs Isaac Lab](../comparisons/mujoco-vs-isaac-lab.md) 与 [仿真物理保真度](../concepts/physics-fidelity-sim2real-gap.md)。
- **单一用途的世界模型** — 多数 WM 只做「预测下一帧」；Pelican-Sim 把同一模型同时当 **数据生成器、策略评测器、动作选择器与策略改进器** 四用，四步 rollout 与 28 维统一动作空间是这四用共享的接口。
- **[Dynin-Robotics](./paper-dynin-robotics.md)（同批）** — 同批中 WM 的另一种摆法：Dynin 把世界建模塞进 **策略骨干内部** 当辅助任务，Pelican-Sim 把它做成 **策略外部的环境替身**；前者服务表征，后者服务闭环。
- **[DATAFARM](./paper-datafarm.md) / [FoldNet++](./paper-foldnet-plus-plus.md)（同批）** — 三条造数据路线的对照：TAMP 规划器（DATAFARM）、仿真渲染（FoldNet++）、世界模型 rollout（本页）。Pelican-Sim 的 50 demo + 500 生成轨迹把成功率从 70% 拉到 93%，是「用生成轨迹补稀缺演示」这一档的代表口径。
- **真机评测** — 文内「策略评测 Pearson **0.994**」是一个 **相关系数**，不是成功率；它说明 WM 给出的排序可信，不等于可以免掉真机验证。相关性的两端具体取哪两组量，归档未写明，以 **原文 PDF** 为准。
- **[RoboTwin](./robotwin.md)** — 本文主实验床；跨页对照 RoboTwin 数字时要确认任务子集与 demo 预算一致，否则不可横比。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **待核实**，仓库可运行性以项目页为准。

## 结论

**Pelican-Sim 1.0 适合作为本期「待核实」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：世界模型价值在于能否被策略训练与决策使用；Pelican-Sim 把 WM 变成数据生成、评测与选动作工具。
2. 开源结论：**待核实** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [pelican-sim_arxiv_2609_12036.md](../../sources/papers/pelican-sim_arxiv_2609_12036.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12036](https://arxiv.org/abs/2609.12036)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12036)
- [项目页](https://zoushilong1024.github.io/Pelican-Sim1.0/)
- [https://github.com/Open-X-Humanoid/Pelican-Sim1.0](https://github.com/Open-X-Humanoid/Pelican-Sim1.0)
