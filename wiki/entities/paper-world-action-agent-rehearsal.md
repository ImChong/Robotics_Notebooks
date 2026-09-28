---
type: entity
tags:
  - paper
  - vla
  - manipulation
  - agent
status: complete
updated: 2026-09-28
arxiv: "2609.29964"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/world-action-agent-rehearsal_arxiv_2609_29964.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "World Action Agent（arXiv:2609.29964）：Visual action workspace；Action Rehearsal 预览修改候选动作；in-view correction；LIBERO-Pro 75.6%。"
---

# World Action Agent（arXiv:2609.29964）

**World Action Agent**（*World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal*，[arXiv:2609.29964](https://arxiv.org/abs/2609.29964)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**Visual action workspace；Action Rehearsal 预览修改候选动作；in-view correction；LIBERO-Pro 75.6%。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- VLM 未在执行前观察动作后果。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.29964](https://arxiv.org/abs/2609.29964) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | WAA with rehearsal + in-view correction + skill accumulation. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- LIBERO-90 技能→LIBERO-Pro/robosuite（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**World Action Rehearsal 把 VLM 变成可预演动作的工作空间。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [world-action-agent-rehearsal_arxiv_2609_29964.md](../../sources/papers/world-action-agent-rehearsal_arxiv_2609_29964.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.29964](https://arxiv.org/abs/2609.29964)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.29964)
