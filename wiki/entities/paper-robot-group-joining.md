---
type: entity
tags: ['paper', 'social-navigation', 'vlm', 'human-robot-interaction', 'navigation']
status: complete
updated: 2026-09-24
arxiv: "2609.28467"
related:
  - ../tasks/vision-language-navigation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
  - ../overview/embodied-13-papers-technology-map.md
sources:
  - ../../sources/papers/robot-group-joining_arxiv_2609_28467.md
  - ../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md
summary: "Robot Group Joining（arXiv:2609.28467）：语言引导预测「社会上合适的加入站位」，而非仅几何路径到固定目标点。"
---

# Robot Group Joining（arXiv:2609.28467）

**Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction**（[项目页](https://robot-join.github.io/)，[arXiv:2609.28467](https://arxiv.org/abs/2609.28467)）来自 [具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）。

## 一句话定义

**语言引导预测「社会上合适的加入站位」，而非仅几何路径到固定目标点。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| TO | Trajectory Optimization | 轨迹优化 |
| HITL | Human-in-the-Loop | 人在回路 |

## 为什么重要

- 人群/队列/观众互动中，加入位姿有社会规范，纯导航目标不足。
- 纳入 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md) 阅读坐标。
- 开源结论（步骤 2.5，2026-09-24）：**待发布**。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28467](https://arxiv.org/abs/2609.28467) |
| **开源** | **待发布** |
| **要点** | 群体成员 grounding + 队形先验 → 多模态加入姿态；静态/动态互动真机验证。 |
| **文内指标** | 静态与动态群体场景真机（文内以 PDF 为准）。 |


## 源码运行时序图

**不适用**（截至 2026-09-24 项目页/论文未提供可运行官方代码；开源状态：**待发布**）。


## 实验与评测

- 静态与动态群体场景真机（文内以 PDF 为准）。
- **读法：** 公众号归纳；逐项 baseline 与协议以 arXiv PDF 为准。

## 与其他工作对比

- 横向索引见 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md)。

## 结论

**加入 = 社会目标预测问题** — 项目页入库日 404，代码待跟进。

1. 开源边界：**待发布** — 以项目页/仓库实际链接为准（入库日 2026-09-24）。
2. 核心机制：群体成员 grounding + 队形先验 → 多模态加入姿态；静态/动态互动真机验证。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [Vision Language Navigation](../tasks/vision-language-navigation.md)
- [Teleoperation](../tasks/teleoperation.md)
- [Vla](../methods/vla.md)
- [Embodied 13 Papers Technology Map](../overview/embodied-13-papers-technology-map.md)

## 参考来源

- [13 篇盘点（公众号）](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)
- [Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction](../../sources/papers/robot-group-joining_arxiv_2609_28467.md)

## 推荐继续阅读

- [arXiv:2609.28467](https://arxiv.org/abs/2609.28467) — 原文
