---
type: entity
tags:
  - paper
  - sim2real
  - locomotion
status: complete
updated: 2026-09-28
arxiv: "2609.28878"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/online-sim2real-closed-loop-modeling_arxiv_2609_28878.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "在线闭环 Sim2Real（arXiv:2609.28878）：把机器人+已部署策略视为闭环系统，学指令–响应关系；在线只改给控制器的参考，不改策略参数。"
---

# 在线闭环 Sim2Real（arXiv:2609.28878）

**在线闭环 Sim2Real**（*Online Sim-to-Real Adaptation via Closed-Loop System Modeling*，[arXiv:2609.28878](https://arxiv.org/abs/2609.28878)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**把机器人+已部署策略视为闭环系统，学指令–响应关系；在线只改给控制器的参考，不改策略参数。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 迁移后残余动力学致持续跟踪偏差。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28878](https://arxiv.org/abs/2609.28878) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Closed-loop system ID on command–response; reference adaptation only. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 双足速度跟踪与移动操作硬件（Duke，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**在线改参考而非重训策略，是轻量 Sim2Real 适应路线。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [online-sim2real-closed-loop-modeling_arxiv_2609_28878.md](../../sources/papers/online-sim2real-closed-loop-modeling_arxiv_2609_28878.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.28878](https://arxiv.org/abs/2609.28878)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.28878)
