---
type: concept
tags: [robotics, dynamics, simulation, sim2real, actuator]
status: complete
updated: 2026-04-27
related:
  - ../methods/beyondmimic.md
  - ../methods/actuator-network.md
  - ./system-identification.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "Armature（电枢惯量 / 反射惯量）是电机转子经过减速器反射到关节侧的等效转动惯量，是机器人高频动态特性建模中不可忽视的项。"
---

# Armature Modeling（电枢惯量建模）

在机器人动力学和仿真中，**Armature** 指的是电机内部旋转部件（转子）的转动惯量，经过减速比放大后，对关节端产生的等效惯性效应。

## 为什么重要

在传统的刚体动力学（如 Pinocchio, Drake 默认配置）中，往往只考虑连杆的质量 and 惯量。但在使用高减速比（如 1:10 以上）的机器人中，电机的转子惯量在关节侧会被放大 **$G^2$** 倍（$G$ 为减速比），甚至可能超过连杆自身的惯量。

- **动态响应**：忽略 Armature 会导致仿真中的关节比现实中表现得更“轻”、响应更“脆”。
- **控制稳定性**：如果仿真中没有建模 Armature，但现实中存在巨大的反射惯量，训练出的策略在真机上容易产生高频振荡。
- **PD 增益设计**：合理的 PD 增益应与关节的总惯量（连杆惯量 + Armature）成比例。

## 物理定义与计算

对于一个典型的执行器系统：
- $J_r$：电机转子的转动惯量。
- $g_1, g_2, \dots$：各级减速比。
- $J_1, J_2, \dots$：中间各级齿轮的转动惯量。

### 单驱动关节
反射到关节端的总等效惯量（Armature）计算公式：

$$
I_{arm} = J_r \cdot (\prod g_i)^2 + \sum (J_k \cdot (\prod_{i>k} g_i)^2)
$$

在大多数工程实践中，简化为：
$$
I_{arm} \approx J_r \cdot G^2
$$

### 双驱动关节（如交叉轴或并联机构）
如果一个自由度由两个完全相同的执行器共同驱动：
$$
I_{arm, total} = 2 \cdot I_{arm, single}
$$

## 在仿真器中的实现

### MuJoCo
在 MJCF 文件中，可以通过 `armature` 属性直接设置：
```xml
<joint name="knee_joint" armature="0.012" ... />
```

### Isaac Gym / Isaac Lab
在 `ArticulatedView` 或机器人描述配置中，可以手动计算并补偿到关节的动态参数中，或者在 `SimConfig` 中通过 `armature` 属性全局或局部设置。

## BeyondMimic 的工程实践

[BeyondMimic](../methods/beyondmimic.md) 强调了精确建模 Armature 是减少 Sim2Real 差距的核心手段：
1. **精确计算**：不再将 armature 视为仿真调参的旋钮，而是通过电机手册数据严格计算。
2. **PD 增益关联**：
   - 设定自然频率 $\omega = 2\pi \times 10$ Hz。
   - 设定阻尼比 $\zeta = 2.0$。
   - 则：$k_p = I_{arm} \cdot \omega^2$, $k_d = 2 \cdot I_{arm} \cdot \zeta \cdot \omega$。

## 关联页面

- [BeyondMimic](../methods/beyondmimic.md) — 强调 armature 精确建模的代表性模仿学习框架。
- [Actuator Network (执行器网络)](../methods/actuator-network.md) — 更复杂的执行器建模方式（如神经网络模拟）。
- [System Identification (系统辨识)](./system-identification.md) — Armature 是系统辨识中的关键物理参数。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- BeyondMimic 技术报告关于物理建模的部分。
