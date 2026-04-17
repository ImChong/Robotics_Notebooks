---
type: concept
tags: [manipulation, contact, force-control, impedance-control, tsid]
status: complete
summary: "Contact-Rich Manipulation 指需要持续建模接触、摩擦和力约束的操作任务，难点不在于碰到物体，而在于控制接触过程本身。"
related:
  - ../tasks/manipulation.md
  - ./contact-dynamics.md
  - ./tsid.md
  - ./whole-body-control.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/contact_planning.md
  - ../../sources/papers/contact_dynamics.md
---

# Contact-Rich Manipulation（接触丰富型操作）

**Contact-Rich Manipulation**：那些必须利用接触力、摩擦、约束和接触序列本身才能完成的操作任务，例如插拔、拧瓶盖、推门、卡扣装配、双手推箱等。

## 一句话定义

这类任务的难点不是“让手碰到物体”，而是“碰到以后如何稳定地利用接触”。

## 为什么重要

非接触抓取更多是在自由空间里规划轨迹；接触丰富型操作则必须面对：
- 法向力与切向力约束
- 接触点切换与滑移
- 任务目标与安全约束冲突
- 视觉误差放大为接触失败

这也是为什么操作从 demo 走向工业部署时，往往卡在接触阶段而不是感知阶段。

## 和普通 manipulation 的区别

| 维度 | 无/弱接触操作 | 接触丰富型操作 |
|------|---------------|----------------|
| 控制重点 | 末端到位、抓取姿态 | 接触力、摩擦、阻抗、约束一致性 |
| 失败模式 | 抓空、对不准 | 卡住、打滑、过力、振荡 |
| 建模难点 | 几何位姿 | 动力学 + 接触模型 |
| 常用执行层 | 位置控制即可 | 往往需要阻抗 / 力控 / WBC / TSID |

## 关键组成

### 1. 接触模型
要知道接触在哪、法向朝哪、接触力是否满足摩擦锥，典型形式是 $\|f_{xy}\| \le \mu f_z$。

### 2. 阻抗 / 力控制
完全刚性的轨迹跟踪在插孔、装配、擦拭等任务里很脆弱；允许一定柔顺性才能吸收模型误差。

### 3. 接触时序
很多任务不是单一接触，而是“接近 → 轻触 → 施力 → 滑动/旋转 → 脱离”的阶段切换问题。

### 4. 全身协调
对人形和移动操作平台来说，手在施力时身体也要提供支撑，任务会自然连到 WBC、TSID 和 loco-manipulation。

## 典型任务

- 插头插座 / peg-in-hole
- 旋钮、门把手、抽屉
- 推箱子、扶墙支撑
- 双手装配、擦拭、打磨

## 与现有页面的关系

- [Contact Dynamics](./contact-dynamics.md) 提供接触力、摩擦锥和约束一致性的物理基础。
- [TSID](./tsid.md) / [Whole-Body Control](./whole-body-control.md) 提供多任务和力约束执行层。
- [Manipulation](../tasks/manipulation.md) 是更上层的任务总览；本页强调其中“最难的接触子域”。

## 常见误区

- **误区 1：接触丰富型操作就是更难的抓取。**
  不完全对。抓取强调是否抓住，contact-rich 更强调持续接触中的力学控制。
- **误区 2：只要视觉足够准，就不需要力控。**
  接触几何误差通常会被放大，纯视觉定位很难替代柔顺执行。
- **误区 3：接触力不需要物理约束。**
  错。摩擦锥、法向非负和接触一致性是执行层的硬边界。

## 参考来源

- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触隐式优化、多接触规划与接触序列组织
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力、摩擦约束与动力学建模基础
- Mordatch et al., *Contact-Invariant Optimization for Hand Manipulations*

## 关联页面

- [Manipulation](../tasks/manipulation.md)
- [Contact Dynamics](./contact-dynamics.md)
- [TSID](./tsid.md)
- [Whole-Body Control](./whole-body-control.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)

## 推荐继续阅读

- Posa et al., *Trajectory Optimization with Discontinuous Contact Dynamics*
- 接触隐式优化 / 阻抗控制综述
- [Query：做机器人操作用模仿学习还是 RL？](../queries/il-for-manipulation.md)
