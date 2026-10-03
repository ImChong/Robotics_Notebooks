---
type: entity
tags:
  - paper
  - humanoid
  - motion-tracking
  - teacher-student
status: complete
updated: 2026-10-03
arxiv: "2609.25754"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/plat-sparse-keyframe-tracking_arxiv_2609_25754.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "PLAT（arXiv:2609.25754）：稠密动作专家→DAgger 学 latent 转移先验→RL 只修正 latent 转移；部署仅需稀疏关键帧+到达时间。"
---

# PLAT（arXiv:2609.25754）

**PLAT**（*PLAT: Sparse Timed Keyframe Motion Tracking for Humanoid Control via Privileged Latent Transition Learning*，[arXiv:2609.25754](https://arxiv.org/abs/2609.25754)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**稠密动作专家→DAgger 学 latent 转移先验→RL 只修正 latent 转移；部署仅需稀疏关键帧+到达时间。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 稠密逐帧跟踪无法作高层运动控制器。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["稠密动作专家"]
    N1["DAgger 蒸馏"]
    N2["潜在转移先验"]
    N3["RL 残差修正"]
    N4["稀疏关键帧与到达时间"]
    N5["部署跟踪"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N5
    N4 --> N5
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.25754](https://arxiv.org/abs/2609.25754) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Privileged latent transition learning; sparse timed keyframes at deploy time. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 稀疏关键帧跟踪部署（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**PLAT 把「改 latent 而非动作残差」用于可稀疏部署的 motion tracking。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [plat-sparse-keyframe-tracking_arxiv_2609_25754.md](../../sources/papers/plat-sparse-keyframe-tracking_arxiv_2609_25754.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.25754](https://arxiv.org/abs/2609.25754)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.25754)
