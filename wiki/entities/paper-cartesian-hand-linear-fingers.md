---
type: entity
tags:
  - paper
  - dexterous-manipulation
  - hardware
status: complete
updated: 2026-09-28
arxiv: "2609.25696"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/cartesian-hand-linear-fingers_arxiv_2609_25696.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "Cartesian Hand（arXiv:2609.25696）：7-DoF 全线性：双平行夹爪+四平移指尖；35 种物体操作；计划开源软硬件。"
---

# Cartesian Hand（arXiv:2609.25696）

**Cartesian Hand**（*The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers*，[arXiv:2609.25696](https://arxiv.org/abs/2609.25696)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**7-DoF 全线性：双平行夹爪+四平移指尖；35 种物体操作；计划开源软硬件。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 仿人灵巧手复杂；平行夹爪无手内操作。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.25696](https://arxiv.org/abs/2609.25696) |
| **开源** | **宣称将开源**（步骤 2.5，2026-09-28） |
| **方法摘要** | All-linear 7-DoF end-effector with composable linear primitives. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 35 种 in-hand 任务；可迁人形双臂（Duke 等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **宣称将开源** — 部署前以项目页/arXiv 为准 |

## 结论

**Cartesian Hand 用线性原语组合实现高覆盖手内操作。**

1. 开源：**宣称将开源**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [cartesian-hand-linear-fingers_arxiv_2609_25696.md](../../sources/papers/cartesian-hand-linear-fingers_arxiv_2609_25696.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.25696](https://arxiv.org/abs/2609.25696)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.25696)
