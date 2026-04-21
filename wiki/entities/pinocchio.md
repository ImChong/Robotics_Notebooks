---
type: entity
tags: [software, dynamics, c++, whole-body-control, algorithms]
status: complete
updated: 2026-04-21
related:
  - ../concepts/whole-body-control.md
  - ../concepts/centroidal-dynamics.md
  - ../concepts/floating-base-dynamics.md
sources:
  - ../../sources/papers/simulation.md
summary: "Pinocchio 是一个基于 C++ 的极致高性能刚体动力学库，是目前各类腿足机器人 WBC 和基于优化的控制器背后的核心计算引擎。"
---

# Pinocchio (刚体动力学库)

**Pinocchio** 是一个由法国国家信息与自动化研究所（INRIA）开源的，专注于**高计算效率**和**分析导数 (Analytical Derivatives)** 的刚体动力学（Rigid Body Dynamics）C++ 库。

在当前的足式机器人和复杂机械臂控制（如 WBC、MPC、DDP）领域，Pinocchio 已经成为了事实上的行业底层标准。

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

## 关联页面
- [Query：Pinocchio 快速上手指南](../queries/pinocchio-quick-start.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)

## 参考来源
- Carpentier, J., et al. (2019). *The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives*.
