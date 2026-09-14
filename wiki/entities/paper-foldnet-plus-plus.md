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

### 数据集速查四维

| 维度 | 归档口径（2026-09-14） |
|------|----------------------|
| **规模** | 6 机器人 × 1K T 恤 × 1K 环境，**120K episode** |
| **模态** | 仿真渲染视觉 + 本体状态 |
| **许可证** | **未知**（待发布） |
| **重定向就绪度** | 生成阶段已按 6 种本体分别渲染 |

- **模态：** 全合成，归档未列出具体通道与分辨率；触觉 / 力反馈无提及。
- **重定向就绪度：** 把跨本体重定向前置到数据生成阶段解决，代价是换到这 6 种以外的本体仍需自行重新生成。


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 120K episode；跨机器人与未见衣物部署测试。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **[FoldEx](./paper-foldex-deformable-clothes-benchmark.md)** — 同为衣物折叠，但分工相反：FoldEx 是 **评测侧** 的可变形衣物基准，FoldNet++ 是 **数据侧** 的合成资产与 episode 生产线；两者可配成「造数据 + 量成绩」一对。
- **[DATAFARM](./paper-datafarm.md)（同批）** — 同为「合成数据换真机数据」，来源不同：DATAFARM 从 **TAMP 规划器** 造轨迹并对齐 VLA 预训练分布，FoldNet++ 从 **仿真渲染** 造跨机器人、跨衣物的 120K episode。
- **[Pelican-Sim 1.0](./paper-pelican-sim.md)（同批）** — 另一条合成数据来源：**世界模型 rollout**。三者构成本期「规划器 / 渲染器 / 世界模型」三种造数据方式的对照。
- **[Sim2Real](../concepts/sim2real.md) 与 [四条路线的可辨识性](../comparisons/sim2real-four-routes-identifiability.md)** — FoldNet++ 押的是 **纯合成训练直接部署**（不做真机微调）这一档；可变形物体的接触与褶皱动力学是该档最吃力的地方，也是本文「跨机器人与未见衣物部署测试」要回答的问题。
- **单本体数据集（多数 [Manipulation](../tasks/manipulation.md) 数据集的默认形态）** — 6 种机器人共享同一批衣物资产，等于把「换本体还能不能用」的重定向问题前置到数据生成阶段解决，而不是留给策略去泛化。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **待发布**，数据集尚不可直接下载复现。

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
