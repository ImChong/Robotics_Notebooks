---
type: entity
tags:
  - paper
  - humanoid
  - teacher-student
  - depth
  - sim2real
status: complete
updated: 2026-09-28
arxiv: "2609.25450"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/redact-robust-perceptive-locomotion_arxiv_2609_25450.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "REDACT（arXiv:2609.25450）：Teacher–Student + 特征遮蔽 + 共识门控：仅用干净仿真深度训练，迁移到未知视觉损坏与森林场景。"
---

# REDACT（arXiv:2609.25450）

**REDACT**（*REDACT: Robust Perceptive Locomotion under Unseen Visual Corruption*，[arXiv:2609.25450](https://arxiv.org/abs/2609.25450)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**Teacher–Student + 特征遮蔽 + 共识门控：仅用干净仿真深度训练，迁移到未知视觉损坏与森林场景。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 真机深度常遇训练未覆盖的损坏；单纯数据增强无法覆盖未知 corruption。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.25450](https://arxiv.org/abs/2609.25450) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Teacher–Student；continual feature masking；conformal-calibrated consensus gating on depth features. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 结构化环境与森林场景迁移（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**REDACT 把「哪些深度特征仍可信」做成可部署门控，适合未知视觉损坏下的感知 locomotion。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [redact-robust-perceptive-locomotion_arxiv_2609_25450.md](../../sources/papers/redact-robust-perceptive-locomotion_arxiv_2609_25450.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.25450](https://arxiv.org/abs/2609.25450)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.25450)
