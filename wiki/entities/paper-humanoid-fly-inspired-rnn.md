---
type: entity
tags:
  - paper
  - humanoid
  - locomotion
  - neuroscience
status: complete
updated: 2026-09-28
arxiv: "2609.27001"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/humanoid-fly-inspired-rnn_arxiv_2609_27001.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "果蝇启发 RNN 控制器（arXiv:2609.27001）：3609 连续神经状态接 G1 仿真；重置/路径替换定位行为来源。"
---

# 果蝇启发 RNN 控制器（arXiv:2609.27001）

**果蝇启发 RNN 控制器**（*Humanoid Locomotion with a Fly-Inspired Recurrent Controller*，[arXiv:2609.27001](https://arxiv.org/abs/2609.27001)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**3609 连续神经状态接 G1 仿真；重置/路径替换定位行为来源。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 生物启发控制器机制难分析。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27001](https://arxiv.org/abs/2609.27001) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Fly-inspired recurrent controller with mechanistic ablations on G1 sim. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 持续运动依赖本体–指令与循环 motor 状态（港科/Zenbot 等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**该工作提供可追踪机制的生物启发 locomotion 分析范式。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [humanoid-fly-inspired-rnn_arxiv_2609_27001.md](../../sources/papers/humanoid-fly-inspired-rnn_arxiv_2609_27001.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.27001](https://arxiv.org/abs/2609.27001)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.27001)
