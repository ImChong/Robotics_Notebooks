---
type: formalization
tags: [dynamics, robotics, aba, rnea, inverse-dynamics, forward-dynamics]
status: complete
updated: 2026-06-23
related:
  - ./lie-group-rigid-body-motions.md
  - ../concepts/floating-base-dynamics.md
  - ../concepts/urdf-robot-description.md
  - ../entities/pinocchio.md
  - ../queries/pinocchio-quick-start.md
  - ../entities/quadruped-control-curriculum.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "ABA 与 RNEA 是求解树状刚体系统正向/逆向动力学的经典 $O(n)$ 算法；四足浮动基扩展下用于计算 M(q)、g(q) 与仿真积分。"
---

# Articulated Body Algorithms（ABA / RNEA）

**树状刚体系统** 的动力学可用 **递归牛顿–欧拉算法（RNEA）** 求逆动力学、用 **铰接体算法（ABA）** 求正向动力学。二者是 Pinocchio、RBDL、Drake 等库的默认内核，四足课程 Project 1 要求手算/验证 **质量矩阵 $M(q)$ 与重力项 $g(q)$**。

## 一句话定义

> **RNEA**：已知 $\ddot{q}$，求所需 $\tau$；**ABA**：已知 $\tau$，求 $\ddot{q}$——均在 $O(n)$ 时间内沿运动学树递归。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RNEA | Recursive Newton–Euler Algorithm | 逆动力学 $\tau = ID(q,\dot q,\ddot q)$ |
| ABA | Articulated Body Algorithm | 正向动力学 $\ddot q = FD(q,\dot q,\tau)$ |
| CRBA | Composite Rigid Body Algorithm | 组装质量矩阵 $M(q)$ |
| FK | Forward Kinematics | 正运动学，动力学递归的前置 |
| ID | Inverse Dynamics | 逆动力学通称 |
| FD | Forward Dynamics | 正向动力学通称 |
| WBC | Whole-Body Control | 常用 RNEA 求期望力矩 |

## 动力学方程

浮动基四足的标准形式：

$$
M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = S^\top \tau + J_c^\top f_c
$$

- $q \in \mathbb{R}^{n_q}$：广义坐标（基座位姿 + 关节角）
- $\tau$：驱动关节力矩；$S$：选择矩阵（基座行常为 0）
- $f_c$：接触外力；欠驱动体现在 **基座行无直接电机**

### 课程验证要点（Project 1）

- 零力矩时基座 **自由落体** → 检查 $g(q)$ 与浮基未约束
- 静止站立 → $\tau \approx g(q)$（忽略摩擦时）
- 可视化 12 关节传感器 vs 仿真状态

## 算法对照

| 算法 | 输入 | 输出 | 复杂度 | 典型用途 |
|------|------|------|--------|---------|
| RNEA | $q,\dot q,\ddot q$ | $\tau$ | $O(n)$ | WBC、轨迹跟踪、重力补偿 |
| ABA | $q,\dot q,\tau$ | $\ddot q$ | $O(n)$ | 仿真积分、预测模型 |
| CRBA | $q$ | $M(q)$ | $O(n^2)$ 或稀疏优化 | 分析惯量、课程作业 |

## 李群积分

基座姿态在 $SO(3)$ 上，数值积分需 **指数映射** 而非欧拉角累加（见 [Lie Group Rigid Body Motions](./lie-group-rigid-body-motions.md)）。Pinocchio 内部处理流形积分；手写仿真器易在此处出错。

## 常见误区

- **混淆 $n_q$ 与 $n_v$**：四元数姿态导致 $n_q = n_v + 1$，算法实现须用 **流形速度** 维数。
- **只用 FK 不做 ID**：无法做重力补偿与力矩前馈，PD 增益难调。

## 关联页面

- [URDF Robot Description](../concepts/urdf-robot-description.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)
- [Pinocchio](../entities/pinocchio.md)
- [TSID Formulation](./tsid-formulation.md)

## 推荐继续阅读

- Featherstone, *Rigid Body Dynamics Algorithms*
- [Modern Robotics](http://hades.mech.northwestern.edu/) — 递归动力学章节
- [Pinocchio Quick Start](../queries/pinocchio-quick-start.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch2 ABA/RNEA 与 Project 1
