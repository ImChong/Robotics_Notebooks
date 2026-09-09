---
type: entity
tags: ['paper', 'manipulation', 'relational', 'object-centric']
status: complete
updated: 2026-09-09
arxiv: "2609.08743"
venue: "ICRA 2026 Beyond Teleoperation Workshop"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/foci_policy_arxiv_2609_08743.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "FOCI Policy（arXiv:2609.08743）：从演示提取紧凑交互片段，用任务相关物体间相对 SE(3) 轨迹表示关系操作；RLBench + 真机 one-shot。"
---

# FOCI Policy

**FOCI Policy**（*Focus on Object-Centric Interactions for Relational Manipulation Policies*，[arXiv:2609.08743](https://arxiv.org/abs/2609.08743)，[项目/代码](https://fitz0401.github.io/foci-page/)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

关系操作的泛化难点在物体间约束而非末端轨迹——FOCI 用变点检测切交互片段，预测实体间相对 SE(3) 而非 gripper 绝对轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOCI | Focus on Object-Centric Interactions | 本文策略框架 |
| SE(3) | Special Euclidean group | 刚体变换群 |
| RLBench | RLBench | 仿真关系操作基准 |
| BC | Behavior Cloning | 演示学习 |

## 为什么重要

- 空间抽象降轨迹方差 + 时间抽象 isolate 关键交互段
- RLBench 每任务 1 demo；真机 cross-gripper /  clutter 组合 motion planning

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08743](https://arxiv.org/abs/2609.08743) |
| **开源** | **未开源** |
| **项目/代码** | [https://fitz0401.github.io/foci-page/](https://fitz0401.github.io/foci-page/) |

## 核心原理

- 空间抽象降轨迹方差 + 时间抽象 isolate 关键交互段
- RLBench 每任务 1 demo；真机 cross-gripper /  clutter 组合 motion planning
- KU Leuven；项目页有视频，无 GitHub URL

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**FOCI Policy 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 未开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [foci_policy_arxiv_2609_08743.md](../../sources/papers/foci_policy_arxiv_2609_08743.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08743](https://arxiv.org/abs/2609.08743)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08743)
- [项目/代码](https://fitz0401.github.io/foci-page/)
