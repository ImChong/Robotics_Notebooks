---
type: entity
tags:
  - paper
  - humanoid
  - wbc
  - safe-rl
status: complete
updated: 2026-09-28
arxiv: "2609.22075"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/limbo-barrier-objectives-wbc_arxiv_2609_22075.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "LIMBO（arXiv:2609.22075）：围绕冻结控制器残差学习状态–动作屏障函数；任务策略训练时内化安全反馈。"
---

# LIMBO（arXiv:2609.22075）

**LIMBO**（*LIMBO: Learning and Internalizing Model-Free Barrier Objectives for Agile and Safe Whole-Body Control*，[arXiv:2609.22075](https://arxiv.org/abs/2609.22075)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**围绕冻结控制器残差学习状态–动作屏障函数；任务策略训练时内化安全反馈。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 解析安全证书难；在线安全滤波增开销。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.22075](https://arxiv.org/abs/2609.22075) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Model-free control barrier on residual actions around frozen controller. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 高 DoF 人形敏捷安全 WBC（Amazon/Caltech 等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**LIMBO 把屏障结构内化进策略，减少在线滤波依赖。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [limbo-barrier-objectives-wbc_arxiv_2609_22075.md](../../sources/papers/limbo-barrier-objectives-wbc_arxiv_2609_22075.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.22075](https://arxiv.org/abs/2609.22075)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.22075)
