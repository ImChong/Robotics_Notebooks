---
type: overview
tags: [topic, topic-physics-fidelity, simulation, physics, dynamics, contact, friction, sim2real, fidelity]
status: complete
updated: 2026-06-28
summary: "仿真物理保真度专题汇总：从几何/URDF 精度 → 刚体动力学算法（ABA/RNEA）→ 接触/摩擦模型 → 执行器模型四层物理保真度的统一入口，串起各层对 sim2real gap 的贡献、建模成本与取舍，收纳分散的动力学/接触/摩擦/可微仿真概念页。"
---

# 仿真物理保真度（专题汇总）

> **图谱专题视图**：本页是知识图谱「⚙️ 物理保真度 (Physics Fidelity)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=physics-fidelity) 筛选时，本节点为汇总锚点。

## 一句话定义

**仿真物理保真度专题** 关注「仿真把物理建到多准」这条**纵向链路**：从**几何/URDF 精度**，经**刚体动力学算法**与**接触/摩擦模型**，到**执行器模型**，逐层分析每层简化各自贡献哪一种 sim2real gap、代价与收益如何取舍。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机迁移工程主线 |
| URDF | Unified Robot Description Format | 机器人几何/惯量描述格式 |
| ABA | Articulated Body Algorithm | O(n) 正向动力学递归 |
| RNEA | Recursive Newton-Euler Algorithm | O(n) 逆动力学递归 |
| LCP | Linear Complementarity Problem | 硬接触互补问题，求导难 |
| DR | Domain Randomization | 域随机化，覆盖未建模残差 |
| SysID | System Identification | 系统辨识，反推物理参数 |

## 为什么重要

- **保真度不是单点而是链路**：几何标定误差会被上层动力学/接触逐级放大，单层最优 ≠ 全链最优。
- **每层补的 gap 不同**：几何补姿态漂移、动力学补长视界发散、接触/摩擦补打滑抖动、执行器补力矩跟踪——投错层等于白投。
- **保真度 × DR × SysID 互补**：先把能建准的建准，再用 DR 覆盖残差，而非用超大 DR 掩盖机理偏差。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| Query | 四层保真度端到端取舍决策树 | [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) |
| 概念 | 保真度 ↔ sim2real gap 因果 | [Physics Fidelity ↔ Sim2Real Gap](../concepts/physics-fidelity-sim2real-gap.md) |
| ① 几何 | URDF 几何/惯量描述 | [URDF 描述](../concepts/urdf-robot-description.md) |
| ② 动力学 | ABA/RNEA 递归算法 | [Articulated Body Algorithms](../formalizations/articulated-body-algorithms.md) |
| ③ 接触/摩擦 | 接触与关节摩擦建模 | [Contact Dynamics](../concepts/contact-dynamics.md) · [Joint Friction Models](../concepts/joint-friction-models.md) |
| ④ 可微/补偿 | 可微仿真与摩擦补偿 | [Differentiable Simulation](../concepts/differentiable-simulation.md) · [Friction Compensation](../concepts/friction-compensation.md) |

## 与其他专题的关系

- **[Sim2Real](./topic-sim2real.md)**：物理保真度是缩小 sim2real gap 的「建准」一侧，与域随机化/系统辨识互补。
- **[全身控制 (WBC)](./topic-wbc.md)**：浮动基/质心动力学是 WBC 的模型底座，保真度直接影响力分配精度。
- **[步态与移动 (Locomotion)](./topic-locomotion.md)**：接触/摩擦保真度决定腿式步态在真机上是否打滑、抖动。

## 关联页面

- [Physics Fidelity ↔ Sim2Real Gap](../concepts/physics-fidelity-sim2real-gap.md)
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Joint Friction Models](../concepts/joint-friction-models.md)
- [Friction Compensation](../concepts/friction-compensation.md)
- [Differentiable Simulation](../concepts/differentiable-simulation.md)
- [URDF 描述](../concepts/urdf-robot-description.md)
- [Articulated Body Algorithms](../formalizations/articulated-body-algorithms.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Procedural Terrain Generation](../concepts/procedural-terrain-generation.md)

## 参考来源

- 本库归纳自 [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md)、[Physics Fidelity ↔ Sim2Real Gap](../concepts/physics-fidelity-sim2real-gap.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`physics-fidelity` 命中规则）
