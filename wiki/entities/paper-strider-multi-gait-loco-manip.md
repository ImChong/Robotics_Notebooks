---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - teacher-student
status: complete
updated: 2026-09-28
arxiv: "2609.23483"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/strider-multi-gait-loco-manip_arxiv_2609_23483.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "STRIDER（arXiv:2609.23483）：AMP 行走 + 3D 落脚专家 + 笛卡尔上肢；LD-PPO 在线 RL + DAgger + Teacher latent 对齐蒸馏统一 Student。"
---

# STRIDER（arXiv:2609.23483）

**STRIDER**（*STRIDER: Stepping-Enabled Multi-Gait Hierarchical 3D Loco-Manipulation Framework for Humanoid Robots*，[arXiv:2609.23483](https://arxiv.org/abs/2609.23483)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**AMP 行走 + 3D 落脚专家 + 笛卡尔上肢；LD-PPO 在线 RL + DAgger + Teacher latent 对齐蒸馏统一 Student。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 速度指令策略难控三维落点；单独踏步策略难与行走/操作统一。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.23483](https://arxiv.org/abs/2609.23483) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Multi-expert + LD-PPO with DAgger and teacher-conditioned latent alignment. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- X-Humanoid 平台分层 loco-manip（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**STRIDER 用 latent 蒸馏把异构专家合成可踏步的多步态 loco-manip 框架。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [strider-multi-gait-loco-manip_arxiv_2609_23483.md](../../sources/papers/strider-multi-gait-loco-manip_arxiv_2609_23483.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.23483](https://arxiv.org/abs/2609.23483)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.23483)
