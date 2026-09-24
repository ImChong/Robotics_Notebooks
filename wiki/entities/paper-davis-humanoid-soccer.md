---
type: entity
tags: ['paper', 'humanoid', 'locomotion', 'active-vision', 'rl']
status: complete
updated: 2026-09-24
arxiv: "2609.28175"
related:
  - ../tasks/humanoid-soccer.md
  - ../methods/reinforcement-learning.md
  - ./unitree-g1.md
  - ../overview/embodied-13-papers-technology-map.md
sources:
  - ../../sources/papers/davis-humanoid-soccer_arxiv_2609_28175.md
  - ../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md
summary: "DAVIS（arXiv:2609.28175）：仅头部深度 + 本体历史 + 低维指令，端到端输出 **25-DoF** PD 目标做人形足球射门/带球，无需运行时检测/规划模块。"
---

# DAVIS（arXiv:2609.28175）

**DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills**（[项目页](https://thusi-lab.github.io/DAVIS/)，[arXiv:2609.28175](https://arxiv.org/abs/2609.28175)）来自 [具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）。

## 一句话定义

**仅头部深度 + 本体历史 + 低维指令，端到端输出 **25-DoF** PD 目标做人形足球射门/带球，无需运行时检测/规划模块。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| TO | Trajectory Optimization | 轨迹优化 |
| HITL | Human-in-the-Loop | 人在回路 |

## 为什么重要

- 足球接触要闭环感知—接近—对齐—击球—恢复，且自运动导致视角剧变；头动是控制问题的一部分。
- 纳入 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md) 阅读坐标。
- 开源结论（步骤 2.5，2026-09-24）：**待发布**。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28175](https://arxiv.org/abs/2609.28175) |
| **开源** | **待发布** |
| **要点** | LightDepthEncoder + HIM 历史；可见性门控辅助几何；GT→prediction annealing + curriculum + AMP 先验；非对称 critic。 |
| **文内指标** | 仿真 + Noetix E1 真机 + 消融（文内；射门/带球分任务定义）。 |


## 源码运行时序图

**不适用**（截至 2026-09-24 项目页/论文未提供可运行官方代码；开源状态：**待发布**）。


## 实验与评测

- 仿真 + Noetix E1 真机 + 消融（文内；射门/带球分任务定义）。
- **读法：** 公众号归纳；逐项 baseline 与协议以 arXiv PDF 为准。

## 与其他工作对比

- 横向索引见 [13 篇技术地图](../overview/embodied-13-papers-technology-map.md)。

## 结论

**深度-only 可承载主动视觉足球技能** — 训练期特权几何勿误当部署输入。

1. 开源边界：**待发布** — 以项目页/仓库实际链接为准（入库日 2026-09-24）。
2. 核心机制：LightDepthEncoder + HIM 历史；可见性门控辅助几何；GT→prediction annealing + curriculum + AMP 先验；非对称 critic。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [Humanoid Soccer](../tasks/humanoid-soccer.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Unitree G1](./unitree-g1.md)
- [Embodied 13 Papers Technology Map](../overview/embodied-13-papers-technology-map.md)

## 参考来源

- [13 篇盘点（公众号）](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)
- [DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills](../../sources/papers/davis-humanoid-soccer_arxiv_2609_28175.md)

## 推荐继续阅读

- [arXiv:2609.28175](https://arxiv.org/abs/2609.28175) — 原文
