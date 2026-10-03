---
type: entity
tags:
  - paper
  - humanoid
  - perception
  - lidar
status: complete
updated: 2026-10-03
arxiv: "2609.23666"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/unipoint-sensor-fusion-locomotion_arxiv_2609_23666.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "UniPoint（arXiv:2609.23666）：360° LiDAR + 双深度→机身点集体素 token；线性自注意力+本体查询；传感退化注入训练单一全地形策略。"
---

# UniPoint（arXiv:2609.23666）

**UniPoint**（*UniPoint: Unified Point-Level Sensor Fusion for Humanoid Locomotion Across Challenging Terrains*，[arXiv:2609.23666](https://arxiv.org/abs/2609.23666)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**360° LiDAR + 双深度→机身点集体素 token；线性自注意力+本体查询；传感退化注入训练单一全地形策略。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 单前向深度覆盖有限；多相机编码成本高。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["360 度 LiDAR"]
    N1["双深度"]
    N2["机身点集体素 token"]
    N3["线性自注意力"]
    N4["本体查询 cross-attention"]
    N5["全地形策略"]
    N6["传感器退化训练"]
    N0 --> N2
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N6 --> N2
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.23666](https://arxiv.org/abs/2609.23666) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Unified point tokens; linear self-attn + proprio cross-attn; sensor dropout. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 复杂地形单策略（浙大/云深处，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**UniPoint 用点级融合使计算量不随传感器数量线性爆炸。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 多传感器融合地形感知在感知栈选型中的位置

## 参考来源

- [unipoint-sensor-fusion-locomotion_arxiv_2609_23666.md](../../sources/papers/unipoint-sensor-fusion-locomotion_arxiv_2609_23666.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.23666](https://arxiv.org/abs/2609.23666)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.23666)
