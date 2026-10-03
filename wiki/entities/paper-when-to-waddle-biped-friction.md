---
type: entity
tags:
  - paper
  - biped
  - locomotion
status: complete
updated: 2026-10-03
arxiv: "2609.21185"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/when-to-waddle-biped-friction_arxiv_2609_21185.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "何时摇摆行走（arXiv:2609.21185）：五执行器双足比较直立 vs 企鹅式躯干侧移；系统改变质心高度与摩擦。"
---

# 何时摇摆行走（arXiv:2609.21185）

**何时摇摆行走**（*When to Waddle: A Comparative Study of Bipedal Torso-Stabilization on Low-Friction Surfaces*，[arXiv:2609.21185](https://arxiv.org/abs/2609.21185)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**五执行器双足比较直立 vs 企鹅式躯干侧移；系统改变质心高度与摩擦。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 低摩擦下地面反力受限，步态选择不明确。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["质心高度"]
    N1["地面摩擦"]
    N2["直立步态"]
    N3["企鹅式躯干侧移"]
    N4["五执行器双足测试"]
    N5["稳定性比较"]
    N0 --> N4
    N1 --> N4
    N2 --> N4
    N3 --> N4
    N4 --> N5
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.21185](https://arxiv.org/abs/2609.21185) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Comparative upright vs penguin waddle gaits on low-friction surfaces. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 低摩擦高质心企鹅步态更优；较高摩擦低质心更优（CMU/NYU，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**低摩擦场景应优先考虑躯干侧移策略而非仅调 PD。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [when-to-waddle-biped-friction_arxiv_2609_21185.md](../../sources/papers/when-to-waddle-biped-friction_arxiv_2609_21185.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.21185](https://arxiv.org/abs/2609.21185)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.21185)
