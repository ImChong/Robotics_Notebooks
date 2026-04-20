---
type: concept
tags: [control, manipulation, impedance-control, force-control, contact-rich, whole-body-control]
status: complete
updated: 2026-04-20
summary: "Impedance Control 通过把末端行为写成质量-弹簧-阻尼关系，让机器人在接触任务中既能跟踪目标又能保持柔顺。"
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/contact_planning.md
related:
  - ./contact-rich-manipulation.md
  - ./whole-body-control.md
  - ./tsid.md
  - ../tasks/manipulation.md
  - ../queries/contact-rich-manipulation-guide.md
---

# Impedance Control（阻抗控制）

**阻抗控制**：不直接要求机器人“精确走到某个位姿”，而是规定当机器人与环境之间出现位置误差或接触力时，系统应该表现出怎样的弹簧-阻尼响应。

## 一句话定义

阻抗控制的核心不是把误差压到零，而是把“误差会产生什么力、以多快衰减”设计清楚。

## 为什么重要

在自由空间轨迹跟踪中，纯位置控制通常足够；但一旦进入插拔、擦拭、装配、拧紧等接触丰富任务，位置误差会被直接放大成碰撞和冲击。阻抗控制通过给末端加入可调柔顺性，让机器人在碰到环境后不是“硬顶”，而是按预期的刚度和阻尼去吸收误差。

这使它成为 contact-rich manipulation、双臂装配、以及 VLA 输出末端位姿目标时最常见的低层执行接口。

## 基本形式

典型任务空间阻抗控制会把末端行为写成：

$$
F = K(x_{des} - x) + D(\dot{x}_{des} - \dot{x})
$$

其中：

- $x_{des}$ 是目标末端位姿
- $x$ 是当前末端位姿
- $K$ 是刚度矩阵，决定“偏了多少要推回去”
- $D$ 是阻尼矩阵，决定“回去时会不会振荡”

再通过雅可比转置或逆动力学把任务空间力映射到关节力矩。

## 什么时候适合用

- 插孔、擦拭、沿表面滑动等需要柔顺接触的任务
- 视觉误差难以完全消除的操作任务
- VLA / BC / Diffusion Policy 输出末端目标位姿，需要低层控制器安全接住
- 双臂共同操持物体，需要吸收微小几何不一致

## 和其他方法的关系

- 相比**纯位置控制**，阻抗控制更适合处理接触误差。
- 相比**纯力控制**，阻抗控制更容易同时保持运动目标和接触柔顺性。
- 在人形系统里，它经常作为 WBC / TSID 中某个任务空间层的执行语义出现。

## 常见误区

- **误区 1：刚度越大越稳定。**  
  刚度太高时，模型误差会直接转成冲击力，反而更容易振荡或损坏硬件。
- **误区 2：阻抗控制只适用于机械臂。**  
  人形上肢、双臂协同甚至全身接触任务都常用阻抗思路。
- **误区 3：有了阻抗控制就不需要接触建模。**  
  阻抗只提供柔顺执行，摩擦锥、法向力和接触切换问题仍需显式考虑。

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力、摩擦与柔顺执行基础
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触任务中的执行层与接触阶段组织
- Hogan, *Impedance Control: An Approach to Manipulation* — 阻抗控制经典工作

## 关联页面

- [Contact-Rich Manipulation](./contact-rich-manipulation.md)
- [Whole-Body Control](./whole-body-control.md)
- [TSID](./tsid.md)
- [Manipulation](../tasks/manipulation.md)
- [Query：接触丰富操作实践指南](../queries/contact-rich-manipulation-guide.md)
