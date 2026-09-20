---
type: entity
tags:
  - paper
  - quadruped
  - locomotion
  - control
status: complete
updated: 2026-09-20
arxiv: "2609.17946"
related:
  - ../tasks/locomotion.md
  - ../entities/go2-motion-imitation.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/fmhp-quadruped-harmonic-policies_arxiv_2609_17946.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md
summary: "FMHP（arXiv:2609.17946）：指令条件傅里叶级数生成关节基准轨迹；状态反馈在线调节偏置、谐波增益、频率与相位；Go2 暂定 3.67 m/s。"
---

# FMHP（arXiv:2609.17946）

**FMHP**（*Feedback-Modulated Harmonic Policies for Quadruped Locomotion*，[arXiv:2609.17946](https://arxiv.org/abs/2609.17946)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**指令条件傅里叶级数生成关节基准轨迹；状态反馈在线调节偏置、谐波增益、频率与相位；Go2 暂定 3.67 m/s。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FMHP | Feedback-Modulated Harmonic Policies | 本文方法 |
| CPG | Central Pattern Generator | 中枢模式发生器 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 参数化谐波轨迹 + 闭环反馈可在保持可解释性的同时提高速度鲁棒性。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.17946](https://arxiv.org/abs/2609.17946) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | Command-conditioned Fourier series base + online feedback modulation. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- Unitree Go2 ~3.67 m/s + load tests（作者报告）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**FMHP 用谐波先验 + 反馈调制平衡四足速度与可调性。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [go2-motion-imitation](../entities/go2-motion-imitation.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [fmhp-quadruped-harmonic-policies_arxiv_2609_17946.md](../../sources/papers/fmhp-quadruped-harmonic-policies_arxiv_2609_17946.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md)
- [arXiv:2609.17946](https://arxiv.org/abs/2609.17946)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.17946)
