# MPC (Model Predictive Control)

聚焦模型预测控制在机器人（特别是腿式/人形）中的理论、工程实现与应用论文。

## 关注问题

- 如何在嵌入式硬件上实现数百赫兹的实时 OCP 求解？
- 如何通过简化模型（SRBD/LIP）降低计算开销？
- 如何保证带约束优化问题的递归可行性与稳定性？
- 如何将 MPC 与低层 WBC 或高层 RL 策略进行耦合？

## 代表性论文

### 理论与综述

- **Mayne et al. (2000)** — *Constrained Model Predictive Control: Stability and Optimality*. 奠定了 MPC 稳定性的理论框架。
- **Williams et al. (2017)** — *Model Predictive Path Integral Control (MPPI)*. 经典的基于采样的随机 MPC 方法。

### 足式机器人应用 (MIT Cheetah 系列)

- **Di Carlo et al. (2018)** — *Dynamic and Robust Legged Locomotion Using a Simplified Model*. 提出了著名的 **Convex MPC**，用于 MIT Mini Cheetah / Cheetah 3。

### 人形与全身控制

- **Wieber (2006)** — *Trajectory Free Linear Model Predictive Control for Stable Walking*. 将预览控制与 ZMP 约束结合，用于双足行走。
- **Sleiman et al. (2021)** — *A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation*. ANYmal 全身控制与移动操作的统一 MPC 框架。

## 关联页面

- [Model Predictive Control (Method)](../../wiki/methods/model-predictive-control.md)
- [MPC 与 WBC 集成 (Concept)](../../wiki/concepts/mpc-wbc-integration.md)
- [Centroidal Dynamics (Concept)](../../wiki/concepts/centroidal-dynamics.md)
- [LIP / ZMP (Concept)](../../wiki/concepts/lip-zmp.md)
- [Model Predictive Path Integral (Method)](../../wiki/methods/mppi.md)
