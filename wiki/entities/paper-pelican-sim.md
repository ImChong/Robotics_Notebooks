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
