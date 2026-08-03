---
type: overview
tags: [hub, hub-contact-force-control, contact, force-control, impedance, admittance, wrench, compliance, hybrid, manipulation]
status: complete
updated: 2026-08-03
summary: "接触力控知识链汇总：从接触感知/估计 → 力旋量表示 → 阻抗/导纳/混合力位控制 → 接触丰富操作策略四层力控闭环的统一入口，串起各层对操作稳定性/安全性的贡献与带宽/刚度/时延取舍，收纳分散的接触估计/力控/力旋量概念页。"
---

# 接触力控（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=contact-manipulation) 与 [路线页](../../roadmap/depth-contact-manipulation.md)。

## 一句话定义

**接触力控知识链** 关注「机器人如何在接触中稳住力」这条**纵向闭环**：从**接触感知/估计**，经**力旋量表示**与**阻抗/导纳/混合力位控制**，到**接触丰富操作策略**，逐层分析每层如何贡献操作稳定性、各自的带宽/刚度/时延成本与典型失败模式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Wrench | Wrench (Force/Torque) | 六维力/力矩，接触作用的统一表示 |
| Impedance | Impedance Control | 以「位移→力」因果调节交互刚度 |
| Admittance | Admittance Control | 以「力→位移」因果，适合刚性环境 |
| Hybrid F/P | Hybrid Force/Position Control | 按方向分解力控与位控子空间 |
| BW | Control Bandwidth | 力控环可达带宽，受时延/刚度约束 |

## 为什么重要

- **闭环而非单点**：感知时延、力旋量估计误差会被上层控制放大，单层最优 ≠ 全环最优。
- **每层补的问题不同**：感知补「接触在哪」、力旋量补「力多大」、阻抗/导纳补「怎么柔顺」、操作策略补「任务怎么走」——投错层等于白投。
- **带宽/刚度/安全三难**：高带宽刚性利于跟踪却牺牲柔顺安全，阻抗与导纳在接触刚度未知时可能失稳，需按接触保真度权衡。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| Query | 四层闭环端到端取舍决策树 | [接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) |
| 概念 | 闭环带宽 ↔ 接触稳定性因果 | [Contact Force Loop Bandwidth](../concepts/contact-force-loop-bandwidth.md) |
| ① 感知/估计 | 接触状态与力估计 | [Contact Estimation](../concepts/contact-estimation.md) |
| ② 力旋量表示 | 六维力/力旋量锥 | [Contact Wrench Cone](../formalizations/contact-wrench-cone.md) |
| ③ 力位控制 | 力控基础与力位混合 | [Force Control Basics](../concepts/force-control-basics.md) · [Impedance Control](../concepts/impedance-control.md) · [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md) |
| ④ 操作策略 | 接触丰富操作与视触觉 | [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) · [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md) |

## 与其他知识链的关系

- **[触觉 (Tactile)](./hub-tactile.md)**：触觉侧重「摸得着」的感知模态，本知识链侧重「稳得住」的控制闭环，二者在感知层交叠、在控制层互补。
- **[物理保真度 (Physics Fidelity)](./hub-physics-fidelity.md)**：接触/摩擦建模是仿真侧的「建准」，本知识链是真机侧的「控稳」，共同决定接触任务能否 sim2real。
- **[抓取 (Grasp)](./hub-grasp.md)**：稳定抓取与灵巧操作依赖力旋量闭环兜底。

## 关联页面

- [接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md)
- [Contact-Rich Manipulation Guide](../queries/contact-rich-manipulation-guide.md)
- [Contact Force Loop Bandwidth](../concepts/contact-force-loop-bandwidth.md)
- [Contact Estimation](../concepts/contact-estimation.md)
- [Force Control Basics](../concepts/force-control-basics.md)
- [Impedance Control](../concepts/impedance-control.md)
- [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)
- [Contact Wrench Cone](../formalizations/contact-wrench-cone.md)

## 参考来源

- 本库归纳自 [接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md)、[Contact Force Loop Bandwidth](../concepts/contact-force-loop-bandwidth.md)、[Impedance Control](../concepts/impedance-control.md)
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`contact-force-control` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[接触动力学核心论文](../../sources/papers/contact_dynamics.md)、[接触规划与接触隐式优化核心论文](../../sources/papers/contact_planning.md)、[机器人控制算法八大体系详解](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md)
