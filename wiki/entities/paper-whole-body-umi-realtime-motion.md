---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - imitation-learning
status: complete
updated: 2026-10-03
arxiv: "2609.22829"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/whole-body-umi-realtime-motion_arxiv_2609_22829.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "Whole-Body UMI（arXiv:2609.22829）：扩散策略预测 UMI 末端轨迹；独立实时全身生成器转参考；G1 异步层级闭环。"
---

# Whole-Body UMI（arXiv:2609.22829）

**Whole-Body UMI**（*Whole-Body UMI: Transferring UMI Manipulation Skills to Humanoid Whole-Body Manipulation via Real-Time Motion Generation*，[arXiv:2609.22829](https://arxiv.org/abs/2609.22829)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**扩散策略预测 UMI 末端轨迹；独立实时全身生成器转参考；G1 异步层级闭环。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- UMI 末端轨迹无法唯一确定全身协调。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["UMI 示范与任务观测"]
    N1["扩散策略"]
    N2["末端轨迹"]
    N3["实时全身运动生成器"]
    N4["全身参考"]
    N5["G1 跟踪执行"]
    N6["本体反馈"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N3
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.22829](https://arxiv.org/abs/2609.22829) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Decouple task diffusion on EE + real-time whole-body motion generator. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- G1 全身 UMI 迁移（浙大/港中文等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**Whole-Body UMI 解耦语义学习与全身协调生成。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [whole-body-umi-realtime-motion_arxiv_2609_22829.md](../../sources/papers/whole-body-umi-realtime-motion_arxiv_2609_22829.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.22829](https://arxiv.org/abs/2609.22829)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.22829)
