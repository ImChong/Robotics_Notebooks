---
type: entity
tags:
  - paper
  - humanoid
  - teleoperation
  - dexterous-manipulation
status: complete
updated: 2026-09-20
arxiv: "2609.18763"
related:
  - ../tasks/teleoperation.md
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/gated-residual-body-hand-coordination_arxiv_2609_18763.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md
summary: "门控残差身–手协调（arXiv:2609.18763）：冻结两模块，仅学有界残差；动作门控分配关节组修正权限，几何奖励门控强调当前交互约束；腕/指尖误差降 39–56%。"
---

# 门控残差身–手协调（arXiv:2609.18763）

**门控残差身–手协调**（*Gated Residual Body-Hand Coordination for Whole-Body Humanoid Teleoperation*，[arXiv:2609.18763](https://arxiv.org/abs/2609.18763)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**冻结两模块，仅学有界残差；动作门控分配关节组修正权限，几何奖励门控强调当前交互约束；腕/指尖误差降 39–56%。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 全身控制 |
| DoF | Degrees of Freedom | 自由度 |
| HRI | Human-Robot Interaction | 人机交互 |

## 为什么重要

- 全身遥操作中身体与手模块独立优化易冲突；残差+门控可在不破坏原模块前提下协调。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.18763](https://arxiv.org/abs/2609.18763) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | Bounded residual on frozen body/hand modules; action gating + geometry reward gating. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- 腕部/指尖误差降低 39.2%–56.3%（作者报告）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**门控残差是全身遥操作的可插拔协调范式，适合已有 body/hand 栈增量部署。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [teleoperation](../tasks/teleoperation.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [whole-body-control](../concepts/whole-body-control.md)

## 参考来源

- [gated-residual-body-hand-coordination_arxiv_2609_18763.md](../../sources/papers/gated-residual-body-hand-coordination_arxiv_2609_18763.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)
- [arXiv:2609.18763](https://arxiv.org/abs/2609.18763)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.18763)
