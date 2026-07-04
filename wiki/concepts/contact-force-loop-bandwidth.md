---
type: concept
tags: [control, contact, force-control, impedance, manipulation, tactile]
status: complete
updated: 2026-06-30
summary: "力控闭环带宽 ↔ 接触稳定性概念页：明示感知时延、控制刚度、接触离散化如何共同决定可达带宽与接触震荡/穿透边界，以及它与阻抗/导纳选型的耦合关系。"
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/contact_planning.md
related:
  - ../queries/contact-wrench-closed-loop.md
  - ./impedance-control.md
  - ./force-control-basics.md
  - ./contact-estimation.md
  - ./hybrid-force-position-control.md
---

# Contact-Force-Loop Bandwidth（力控闭环带宽 ↔ 接触稳定性）

**力控闭环带宽**：在与刚性环境接触时，力控制器能稳定跟踪力指令的最高频率。它不是一个可以随意调大的旋钮——它被**感知时延、控制刚度、接触离散化**三者联合钳住，超过这条边界，接触就会震荡或穿透。

## 一句话定义

带宽不是「越高越好」，而是「**在当前感知质量与接触刚度下，最高能闭多快的环而不失稳**」——它是 [接触力旋量闭环](../queries/contact-wrench-closed-loop.md) 里①感知层与③控制层之间那条最硬的约束。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Wrench | Force-Torque Wrench | 6D 力旋量 $[\mathbf{f},\boldsymbol{\tau}]$，统一描述接触力与力矩 |
| F/T | Force/Torque Sensor | 力/力矩传感器，提供 6D 接触力旋量测量 |
| ZOH | Zero-Order Hold | 零阶保持，离散控制中两拍之间维持上一指令 |
| SEA | Series Elastic Actuator | 串联弹性执行器，靠弹性元件换取力控柔顺与可测性 |
| BW | Bandwidth | 带宽，闭环可稳定跟踪的频率上限 |

## 为什么重要

接触力控的不稳定几乎都能归到「带宽开太高」这一条上。原因是：机器人末端与刚性环境构成一个闭环，环境刚度 $K_e$ 越大，闭环极点越容易被推到不稳定区。此时唯一能换回稳定裕度的，要么是**降控制刚度**，要么是**降带宽**。理解这条边界，才能解释为什么「环境越硬、阻抗刚度反而要调得越低」这个反直觉的经验法则（见 [Impedance Control](./impedance-control.md)）。

## 三个共同决定带宽的因素

| 因素 | 怎么压低可达带宽 | 工程对策 |
|------|----------------|---------|
| **感知时延** $\tau_d$ | 估计的力旋量滞后于真实接触，等效相位裕度损失 | 提高 F/T / 触觉采样率，缩短估计窗口（见 [Contact Estimation](./contact-estimation.md)） |
| **控制刚度** $K_d$ | 与刚性环境 $K_e$ 串联，刚度越高闭环极点越靠近不稳定 | 在未知接触处主动降刚度，换稳定裕度 |
| **接触离散化** | ZOH 把连续接触切成定步长，步长内力被高估，等效注入能量 | 减小积分/控制步长，或用软接触/SEA 平滑 |

> 经验关系（量级，非精确公式）：可达带宽大致正比于 $1/\tau_d$ 并随 $\sqrt{K_d/m}$ 上升，但一旦 $K_d$ 逼近使闭环极点越界的临界值就必须回退。三者中**任何一个最差的那个**决定上限——这是一条「短板约束」。

## 与阻抗/导纳选型的耦合

- **阻抗控制**（力矩控机器人）天然能跑更高带宽，因为它直接调电机力矩、少了一层位置内环；代价是对感知时延更敏感。
- **导纳控制**（位置控机器人 + 外接 F/T）受限于位置内环带宽，结构上就跑不快，但对噪声更宽容。
- 选型本质是在「带宽需求 vs 感知质量」之间做权衡：**感知干净就上阻抗冲带宽，感知脏就退回导纳保稳健**。这条选型逻辑在 [Force Control Basics](./force-control-basics.md) 与 [Hybrid Force-Position Control](./hybrid-force-position-control.md) 里有更细的子空间划分。

## 接触震荡 vs 接触穿透两条边界

- **震荡边界**：带宽/刚度开太高 + 感知时延 → 末端在接触面上高频抖动，力指令发散。
- **穿透边界**：带宽/刚度开太低或步长太大 → 接触检测不及时，末端「穿」进环境再被弹回，表现为冲击力尖峰。
- 健康的工作点夹在两者之间，且随环境刚度移动——这正是为什么力控调参不能一套参数走天下。

## 常见误区

- **误区 1：采样率拉满就能提带宽。** 采样率只解决离散化与时延，若控制刚度已逼近临界，提采样率也救不回稳定裕度。
- **误区 2：带宽越高接触越稳。** 恰恰相反，刚性接触下盲目提带宽是头号失稳来源；稳定的接触往往来自「克制的带宽 + 匹配的刚度」。

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力、环境刚度与闭环稳定性基础
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触约束与离散化对力控的影响

## 关联页面

- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 本概念在四层闭环中的定位（①↔③ 约束）
- [Impedance Control（阻抗控制）](./impedance-control.md) — 「环境越硬刚度越低」的稳定性来源
- [Force Control Basics（力控制基础）](./force-control-basics.md) — 阻抗/导纳带宽差异的理论背景
- [Contact Estimation（接触估计）](./contact-estimation.md) — 感知时延这一项的来源
- [Hybrid Force-Position Control（力位混合控制）](./hybrid-force-position-control.md) — 子空间划分下的带宽分配
- 专题汇总：[接触力控（专题汇总）](../overview/topic-contact-force-control.md) — 「🤝 接触力控」[图谱专题视图](../../docs/graph.html?topic=contact-force-control)的统一入口
