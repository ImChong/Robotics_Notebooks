---
type: overview
tags: [topic, topic-tactile, haptic, force, contact, visuo-tactile]
status: complete
updated: 2026-06-17
summary: "触觉与力觉闭环专题汇总：覆盖触觉传感、视触觉融合、阻抗/力控与接触估计，强调「摸得着」对抓取与 loco-manip 稳定性的作用。"
---

# 触觉与力觉（专题汇总）

> **图谱专题视图**：本页是知识图谱「✋ 触觉 (Tactile)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=tactile) 筛选时，本节点为汇总锚点。

## 一句话定义

**触觉专题** 研究机器人如何通过 **力、触觉与接触状态** 闭环调节交互，使抓取、装配与 loco-manip 在不确定接触下仍稳定可控。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Tactile | Tactile Sensing | 表面/接触力学感知 |
| Haptic | Haptic Feedback | 力反馈与遥操作触觉 |
| Impedance | Impedance Control | 力-位关系调节的交互控制 |
| Visuo-Tactile | Visuo-Tactile Fusion | 视觉与触觉联合表征 |
| Wrench | Wrench (Force/Torque) | 六维力/力矩测量 |

## 为什么重要

- **视觉有盲区**：遮挡、反光、透明物体下，触觉是接触真相。
- **硬位置控制在接触瞬间易崩**：需要阻抗/力控与接触估计。
- **V21 专题主线**：从 GelSight 类传感器到全身 loco-manip 的力闭环。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 传感 | 触觉模态与硬件 | [Tactile Sensing](../concepts/tactile-sensing.md) |
| 融合 | 视+触联合 | [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md) |
| 控制 | 阻抗与力控基础 | [Impedance Control](../concepts/impedance-control.md)、[Force Control Basics](../concepts/force-control-basics.md) |
| 估计 | 接触状态/力 | [Contact Estimation](../concepts/contact-estimation.md) |
| 执行 | 力位混合 | [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md) |

## 与其他专题的关系

- **[抓取](./topic-grasp.md)**：稳定抓取依赖力闭环。
- **[WBC](./topic-wbc.md)**：全身力分配与接触约束。
- **[通信协议](./topic-communication.md)**：EtherCAT 等低延迟总线服务力控环路。

## 关联页面

- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Safe Real-World RL Fine-Tuning](../concepts/safe-real-world-rl-fine-tuning.md)（接触安全）

## 参考来源

- 本库归纳自 [Tactile Sensing](../concepts/tactile-sensing.md)、[Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)、[Impedance Control](../concepts/impedance-control.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`tactile` 命中规则）
