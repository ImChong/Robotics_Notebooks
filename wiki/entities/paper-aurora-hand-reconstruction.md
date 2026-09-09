---
type: entity
tags: ['paper', 'in-hand-manipulation', 'reconstruction', 'active-perception']
status: complete
updated: 2026-09-09
arxiv: "2609.08493"
venue: "arXiv 2026"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/aurora_hand_reconstruction_arxiv_2609_08493.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "AURORA（arXiv:2609.08493）：Ray-GPIS 不确定性驱动 next-best-view + 轴条件手内旋转；30s 预算 mean F@10=0.9671，优于开环基线。"
---

# AURORA

**AURORA**（*Active Uncertainty-Driven Re-Orientation for In-Hand Reconstruction*，[arXiv:2609.08493](https://arxiv.org/abs/2609.08493)，[项目/代码](https://aurorahand.github.io/)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

固定相机看手内物体时手和物互相遮挡——AURORA 用重建不确定性选下一最佳视角，再用手内旋转策略主动暴露未观测表面。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AURORA | Active Uncertainty-Driven Re-Orientation | 本文主动重建框架 |
| NBV | Next Best View | 下一最佳视角规划 |
| RGB-D | RGB-Depth | 固定相机观测 |
| F-score | F-score | 重建精度/召回调和均值 |

## 为什么重要

- Leap Hand + BundleTrack + 视觉融合管线
- 30s 操作预算：online F@10 mean 0.9671，active 优于 x/y/z 单轴与固定 schedule

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08493](https://arxiv.org/abs/2609.08493) |
| **开源** | **未开源** |
| **项目/代码** | [https://aurorahand.github.io/](https://aurorahand.github.io/) |

## 核心原理

- Leap Hand + BundleTrack + 视觉融合管线
- 30s 操作预算：online F@10 mean 0.9671，active 优于 x/y/z 单轴与固定 schedule
- 匿名作者项目页；无公开代码链

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**AURORA 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 未开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [aurora_hand_reconstruction_arxiv_2609_08493.md](../../sources/papers/aurora_hand_reconstruction_arxiv_2609_08493.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08493](https://arxiv.org/abs/2609.08493)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08493)
- [项目/代码](https://aurorahand.github.io/)
