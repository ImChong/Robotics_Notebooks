---
type: formalization
tags: [locomotion, control, dynamics, humanoid, math]
status: complete
updated: 2026-04-20
related:
  - ../concepts/lip-zmp.md
  - ../concepts/centroidal-dynamics.md
  - ../methods/model-predictive-control.md
  - ../tasks/locomotion.md
  - ./friction-cone.md
sources:
  - ../../sources/papers/mpc.md
summary: "ZMP + LIP（Linear Inverted Pendulum）模型通过简化质心高度和角动量，将人形机器人平衡问题转化为可解析的线性动力学方程。"
---

# ZMP + LIP 形式化

**LIP (Linear Inverted Pendulum)** 模型是将人形机器人简化为在固定高度平面内运动的质点，而 **ZMP (Zero Moment Point)** 则是判断该系统是否平衡的关键指标。

## 核心假设

1. **质心高度恒定**：质心（CoM）在 $z$ 方向上的位置固定为 $z_c$。
2. **忽略角动量**：忽略由于躯干摆动产生的角动量变化 $\dot{L}_G \approx 0$。
3. **单支撑脚点接触**：假设支撑脚为理想点接触或刚性平面接触。

## LIP 动力学方程

在这些假设下，质心在水平面（$x, y$ 方向）上的运动方程变得完全解耦且线性：

$$
\ddot{x} = \omega^2 (x - x_{zmp})
$$
$$
\ddot{y} = \omega^2 (y - y_{zmp})
$$

其中：
- $\omega = \sqrt{g / z_c}$ 是倒立摆的固有频率。
- $g$ 是重力加速度。
- $(x, y)$ 是质心位置。
- $(x_{zmp}, y_{zmp})$ 是零力矩点（ZMP）位置。

## ZMP 稳定性条件

ZMP 是地面对机器人反作用力合力作用点。机器人保持动态平衡的**充分必要条件**（在 LIP 假设下）是：

$$
\mathbf{p}_{zmp} \in \mathcal{S}_{support\_polygon}
$$

即 ZMP 必须落在支撑多边形（Support Polygon）内。如果 ZMP 移动到边缘，系统将面临翻倒。

## 捕获点（Capture Point, CP）

基于 LIP 方程，可以导出**捕获点（Capture Point）**。CP 是机器人必须跨步才能停止的位置：

$$
\mathbf{\xi} = \mathbf{x} + \frac{1}{\omega} \dot{\mathbf{x}}
$$

当 CP 位于支撑多边形外时，机器人如果不跨步，即使 ZMP 落在支撑区内也无法最终停止，必然会跌倒。

## 在控制中的应用

1. **预览控制（Preview Control）**：通过给定的足迹序列预先规划 ZMP 轨迹，再通过求解上述线性差分方程获得最优 CoM 轨迹。
2. **DCM 控制（Divergent Component of Motion）**：将 CoM 运动分解为稳定分量和不稳定分量（即 CP），通过控制 CP 的运动来实现实时平衡。
3. **线性 MPC**：将上述方程作为预测模型，在 QP 求解器中施加 ZMP 边界约束，实现鲁棒步行。

## 关联页面
- [LIP / ZMP 概念](../concepts/lip-zmp.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Model Predictive Control](../methods/model-predictive-control.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Friction Cone](./friction-cone.md)

## 参考来源
- Kajita, S., et al. (2001). *The 3D Linear Inverted Pendulum Mode: A simple modeling for a biped robot*.
- Wieber, P.-B. (2006). *Trajectory Free Linear Model Predictive Control for Stable Walking*.
