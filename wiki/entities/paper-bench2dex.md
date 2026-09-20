---
type: entity
tags:
  - paper
  - benchmark
  - dexterous-manipulation
  - bimanual
  - vla
status: complete
updated: 2026-09-20
arxiv: "2609.15726"
related:
  - ../entities/isaac-lab.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/bench2dex_arxiv_2609_15726.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md
summary: "Bench2Dex（arXiv:2609.15726）：Isaac Lab 统一 visuotactile 双臂基准：12 种灵巧手、26 任务、~1300 遥操作 demo；7 类扰动分 invariance/equivariance；评测 ACT/DP/π0.5/GR00T N1.5。"
---

# Bench2Dex（arXiv:2609.15726）

**Bench2Dex**（*Bench2Dex: Benchmarking Visuo-Tactile Bimanual Dexterous Manipulation Across Dexterous Hands*，[arXiv:2609.15726](https://arxiv.org/abs/2609.15726)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**Isaac Lab 统一 visuotactile 双臂基准：12 种灵巧手、26 任务、~1300 遥操作 demo；7 类扰动分 invariance/equivariance；评测 ACT/DP/π0.5/GR00T N1.5。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作 |
| BC | Behavior Cloning | 行为克隆 |
| DP | Diffusion Policy | 扩散策略 |

## 为什么重要

- 跨手型灵巧操作缺统一 visuotactile 双臂基准与扰动轴。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.15726](https://arxiv.org/abs/2609.15726) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | Unified Isaac Lab benchmark + human demos + perturbation axes + multi-policy eval. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- 12 hands, 26 tasks, ~1300 demos; ACT/DP/π0.5/GR00T N1.5 comparison.
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**Bench2Dex 为跨灵巧手 visuotactile 双臂策略提供可复现横评底座。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [isaac-lab](../entities/isaac-lab.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [bench2dex_arxiv_2609_15726.md](../../sources/papers/bench2dex_arxiv_2609_15726.md)
- [wechat_senlanke_weekly_manipulation_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)
- [arXiv:2609.15726](https://arxiv.org/abs/2609.15726)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.15726)
