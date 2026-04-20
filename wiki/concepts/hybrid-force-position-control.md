---
type: concept
tags: [control, manipulation, force-control, position-control, contact-rich, assembly]
status: complete
updated: 2026-04-20
summary: "力位混合控制把不同方向上的控制目标拆分为位置跟踪与力跟踪两类，是接触丰富操作中最常见的执行层结构之一。"
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/contact_planning.md
related:
  - ./impedance-control.md
  - ./contact-rich-manipulation.md
  - ./whole-body-control.md
  - ../tasks/manipulation.md
  - ../queries/contact-rich-manipulation-guide.md
---

# Hybrid Force-Position Control（力位混合控制）

**力位混合控制**：把任务空间拆成“该控位置的方向”和“该控力的方向”，让机器人在一个子空间内严格跟踪几何目标，在另一个子空间内稳定施加期望接触力。

## 一句话定义

它不是在同一方向同时强行控位置又控力，而是先决定**哪些自由度听位置，哪些自由度听力**。

## 为什么重要

很多接触任务如果只做位置跟踪，会把环境误差直接转换成过大接触力；如果只做力控，又会失去几何精度。力位混合控制提供了一个更实用的折中：

- 法向方向控力，避免撞击过猛
- 切向方向控位置，沿着目标轨迹移动
- 适合装配、拧紧、擦拭、沿边探索等任务

## 典型例子

### 1. 插孔

- 插入方向：控力或低刚度
- 横向方向：控位置
- 目标是边接触边纠偏，而不是生硬把末端塞进去

### 2. 擦拭

- 法向：维持稳定压力
- 平面内：沿轨迹移动

### 3. 旋钮/螺丝

- 轴向：施加预紧力
- 旋转自由度：控角度或速度

## 基本形式

设任务空间误差为 $e_x$，接触力误差为 $e_f$，通常构造两个选择矩阵：

$$
S_p + S_f = I
$$

其中：

- $S_p$ 选出做位置控制的方向
- $S_f$ 选出做力控制的方向

控制律常写成：

$$
u = S_p u_{pos} + S_f u_{force}
$$

核心不是公式复杂，而是**任务拆分是否合理**。如果拆分错了，系统就会在接触边界反复打架。

## 与阻抗控制的关系

- **阻抗控制**更像让系统表现成“有弹簧阻尼的柔顺体”
- **力位混合控制**更强调任务方向拆分和目标分配

很多工程系统会把两者结合起来：位置子空间里做阻抗跟踪，力子空间里做显式力反馈。

## 何时使用

适合：

- 接触方向明确的任务
- 需要稳定法向压力但仍需几何跟踪的任务
- 环境表面法向可以可靠估计的场景

不太适合：

- 接触法向经常变化且难以实时估计
- 任务需要非常复杂的多点接触切换
- 双臂闭链里同时存在强耦合内力约束但没有统一优化层

## 常见误区

- **误区 1：所有方向都能同时做高精度力控和位置控。**  
  真实系统里通常必须做方向拆分，否则会出现约束冲突。
- **误区 2：一旦用了力位混合控制就不需要阻抗。**  
  在噪声和延迟存在时，柔顺性仍然很重要。
- **误区 3：法向方向永远固定。**  
  接触几何一变，选择矩阵也可能需要重新定义。

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力、约束与摩擦锥基础
- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触阶段和任务空间约束组织
- Raibert & Craig, *Hybrid Position/Force Control of Manipulators*

## 关联页面

- [Impedance Control](./impedance-control.md)
- [Contact-Rich Manipulation](./contact-rich-manipulation.md)
- [Whole-Body Control](./whole-body-control.md)
- [Manipulation](../tasks/manipulation.md)
- [Query：接触丰富操作实践指南](../queries/contact-rich-manipulation-guide.md)
