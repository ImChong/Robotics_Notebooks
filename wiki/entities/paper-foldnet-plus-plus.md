---
type: entity
tags:
  - paper
  - dataset
  - deformable
  - manipulation
  - sim2real
status: complete
updated: 2026-09-14
arxiv: "2609.12433"
related:
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ./paper-datafarm.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/foldnet-plus-plus_arxiv_2609_12433.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "6 种机器人、1K T 恤、1K 环境资产、120K episode 合成数据；测试纯合成训练的视觉运动策略跨机器人与未见衣物部署。"
---

# FoldNet++（arXiv:2609.12433）

**FoldNet++**（[FoldNet++: a Large-Scale Synthetic Dataset for Robotic T-Shirt Folding and Unfolding](https://arxiv.org/abs/2609.12433)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。衣物大变形使真实采集昂贵、Sim2Real 难；FoldNet++ 提供带关键点与子任务标注的大规模合成集。

## 一句话定义

**6 种机器人、1K T 恤、1K 环境资产、120K episode 合成数据。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 衣物大变形使真实采集昂贵、Sim2Real 难；FoldNet++ 提供带关键点与子任务标注的大规模合成集。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12433](https://arxiv.org/abs/2609.12433) |
| **项目页** | https://pku-epic.github.io/FoldNetXX/ |
| **开源** | **待发布** |
| **文内指标** | 120K episode；跨机器人与未见衣物部署测试。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 120K episode；跨机器人与未见衣物部署测试。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 结论

**FoldNet++ 适合作为本期「待发布」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：衣物大变形使真实采集昂贵、Sim2Real 难；FoldNet++ 提供带关键点与子任务标注的大规模合成集。
2. 开源结论：**待发布** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [foldnet-plus-plus_arxiv_2609_12433.md](../../sources/papers/foldnet-plus-plus_arxiv_2609_12433.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12433](https://arxiv.org/abs/2609.12433)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12433)
- [项目页](https://pku-epic.github.io/FoldNetXX/)
