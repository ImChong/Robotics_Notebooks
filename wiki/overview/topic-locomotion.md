---
type: overview
tags: [topic, topic-locomotion, gait, bipedal, walking, humanoid]
status: complete
updated: 2026-06-17
summary: "Locomotion 步态专题汇总：双足/人形/四足在不同地形上的稳定移动，覆盖步态生成、ZMP/LIP、MPC 与 RL 路线及感知式越障。"
---

# Locomotion 步态（专题汇总）

> **图谱专题视图**：本页是知识图谱「🚶 Locomotion」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=locomotion) 筛选时，本节点为汇总锚点。

## 一句话定义

**Locomotion** 研究腿式与人形机器人如何在 **平地、楼梯与复杂地形** 上稳定、高效地移动，是运动控制最基础也最难的任务之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco | Locomotion | 移动/行走能力 |
| ZMP | Zero Moment Point | 经典稳定性判据 |
| LIP | Linear Inverted Pendulum | 简化质心动力学模型 |
| DCM | Divergent Component of Motion | 与捕获点相关的稳定性量 |
| MPC | Model Predictive Control | 预测式步态/质心规划 |

## 为什么重要

- **人形首要能力**：不能稳定走，上层 loco-manip 无从谈起。
- **方法谱系广**：从 ZMP 经典控制到 RL 端到端，再到 MPC+WBC 混合。
- **Sim2Real 高发区**：接触与地形随机性导致迁移难度高。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 任务 | Locomotion 总览 | [Locomotion](../tasks/locomotion.md) |
| 步态 | 相位与生成 | [Gait Generation](../concepts/gait-generation.md) |
| 模型 | LIP / ZMP / 捕获点 | [LIP-ZMP](../concepts/lip-zmp.md)、[Capture Point / DCM](../concepts/capture-point-dcm.md) |
| 规划 | 落足与地形 | [Footstep Planning](../concepts/footstep-planning.md)、[Terrain Adaptation](../concepts/terrain-adaptation.md) |
| 越障 | 楼梯/跑酷索引 | [Stair & Obstacle Locomotion](../tasks/stair-obstacle-perceptive-locomotion.md) |

## 与其他专题的关系

- **[WBC](./topic-wbc.md)**：执行层协调全身满足行走约束。
- **[Sim2Real](./topic-sim2real.md)**：行走策略迁移是典型难题。
- **[状态估计](./topic-state-estimation.md)**：感知式 locomotion 依赖里程计/地形估计。

## 关联页面

- [Model Predictive Control](../methods/model-predictive-control.md)
- [MPC vs RL](../comparisons/mpc-vs-rl.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)

## 参考来源

- 本库归纳自 [Locomotion 任务页](../tasks/locomotion.md) 及相关概念/方法页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`locomotion` 命中规则）
