---
type: entity
tags:
  - paper
  - humanoid
  - motion-generation
status: complete
updated: 2026-09-28
arxiv: "2609.22611"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/higennto-noise-space-optimization_arxiv_2609_22611.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "HIGenNTO（arXiv:2609.22611）：优化预训练文本动作模型的初始噪声而非动作序列，满足接触/避碰/支撑约束。"
---

# HIGenNTO（arXiv:2609.22611）

**HIGenNTO**（*HIGenNTO: Scalable Humanoid Interaction Generation via Noise-Space Trajectory Optimization*，[arXiv:2609.22611](https://arxiv.org/abs/2609.22611)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**优化预训练文本动作模型的初始噪声而非动作序列，满足接触/避碰/支撑约束。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 接触丰富交互难获稳定物理可执行参考。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.22611](https://arxiv.org/abs/2609.22611) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Noise-space optimization on generative motion prior with physics constraints. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 可训练跟踪器或深度视觉策略（CMU/庆应，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**HIGenNTO 在噪声空间做约束优化，扩展接触交互参考生成。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [higennto-noise-space-optimization_arxiv_2609_22611.md](../../sources/papers/higennto-noise-space-optimization_arxiv_2609_22611.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.22611](https://arxiv.org/abs/2609.22611)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.22611)
