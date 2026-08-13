---

type: entity
tags: [software, dynamics, c++, whole-body-control, algorithms, inria]
status: complete
updated: 2026-08-13
related:
  - ../concepts/whole-body-control.md
  - ../concepts/centroidal-dynamics.md
  - ../concepts/floating-base-dynamics.md
  - ./paper-urdd-universal-robot-description-directory.md
  - ./dynibo.md
  - ../methods/joint-actuator-parameter-identification.md
  - ./flobaroid.md
  - ../formalizations/forward-kinematics.md
  - ../formalizations/robot-jacobian.md
sources:
  - ../../sources/papers/simulation.md
  - ../../sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md
  - ../../sources/repos/dynibo.md
summary: "Pinocchio 是一个基于 C++ 的极致高性能刚体动力学库，是目前各类腿足机器人 WBC 和基于优化的控制器背后的核心计算引擎。"
---

# Pinocchio (刚体动力学库)

**Pinocchio** 是一个由法国国家信息与自动化研究所（INRIA）开源的，专注于**高计算效率**和**分析导数 (Analytical Derivatives)** 的刚体动力学（Rigid Body Dynamics）C++ 库。

在当前的足式机器人和复杂机械臂控制（如 WBC、MPC、DDP）领域，Pinocchio 已经成为了事实上的行业底层标准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| iLQR | iterative Linear Quadratic Regulator | 对非线性系统迭代线性化求解的轨迹优化方法 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| DoF | Degrees of Freedom | 自由度，人形通常 20–50+ 关节 |

## 核心特性

1. **极致的性能**：
   Pinocchio 采用了基于模板元编程的架构（利用 Eigen 库），避免了运行时的动态内存分配。这使得它的正向/逆向动力学计算（如 Featherstone 算法）和雅可比求值的速度远远超过其他同类库（如 RBDL 或 KDL）。在 1000Hz 甚至更高频的控制环路中，性能至关重要。
2. **解析导数支持**：
   现代控制算法（如 iLQR 或 DDP）极度依赖动力学的偏导数（即 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial u}$）。Pinocchio 原生提供了这些偏导数的极速解析计算接口，这也是它能垄断最优化控制底层框架的核心原因。
3. **浮动基座与质心动力学**：
   原生支持六自由度浮动基座（Floating Base）的运动学和动力学建模，并提供了专用的接口计算质心动量矩阵（Centroidal Momentum Matrix, CMM）和非线性偏置力，这对于双足/四足机器人控制极为友好。

## 典型技术栈组合

- **Pinocchio + OSQP/qpOASES**：构成经典的 Whole-Body Control (WBC) 控制器底座。
- **Pinocchio + Crocoddyl**：构成目前最高效的差分动态规划（DDP）和全身 MPC 求解器框架。

## 与 URDD 的分工

[URDD](./paper-urdd-universal-robot-description-directory.md)（arXiv:2512.23135）把各框架从 URDF **重复派生** 的 DOF 映射、链结构等 **模块化落盘**；Pinocchio 负责 **给定模型后的动力学计算**——二者正交，URDD 是 **进 Pinocchio 之前的共享预处理层**。

## 与 Dynibo 的对照

[Dynibo](./dynibo.md)（Rust，MIT，v0.1.0）聚焦 **树状 URDF + Workspace 零分配** 的常用子集（FK / Jacobian / DLS-IK / 重力 / RNEA），并以 Pinocchio 作 **oracle 与 Criterion 对照**。需要解析导数、浮动基质心动量、ABA/CRBA 或 Crocoddyl 生态时仍选 Pinocchio；只需轻量多语言内核时可评估 Dynibo。

## 动力学回归矩阵

`pin.computeJointTorqueRegressor(model, data, q, v, a)` 给出连杆 10 参数的 $Y_{\mathrm{rb}}$，使 $\tau = Y_{\mathrm{rb}}\pi_{\mathrm{rb}}$。它 **不含** armature / 粘滞 / 库仑列；关节执行器参数要在 $Y$ 上自拼 $\ddot q_i$、$\dot q_i$、$\mathrm{sign}(\dot q_i)$。完整估法见 [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md)；有力矩传感的流水线对照 [FloBaRoID](./flobaroid.md)（动力学核是 iDynTree）。

## 关联页面
- [Query：Pinocchio 快速上手指南](../queries/pinocchio-quick-start.md)
- [Dynibo](./dynibo.md) — Rust 轻量 FK/RNEA/数值 IK，Pinocchio oracle 对照
- [正向运动学](../formalizations/forward-kinematics.md) — URDF 树 FK 的教学对照
- [雅可比矩阵](../formalizations/robot-jacobian.md) — `computeFrameJacobian` 几何雅可比
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)
- [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) — `computeJointTorqueRegressor` 只给 $Y_{\mathrm{rb}}$
- [FloBaRoID](./flobaroid.md) — 线性辨识流水线（iDynTree，非本库）

## 参考来源
- Carpentier, J., et al. (2019). *The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives*.
- [sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md](../../sources/papers/urdd_beyond_urdf_arxiv_2512_23135.md) — URDD 与 Pinocchio 等「从模型描述推导动力学」栈的交叉引用
- [sources/repos/dynibo.md](../../sources/repos/dynibo.md) — Dynibo 与 Pinocchio 性能/oracle 对照归档
