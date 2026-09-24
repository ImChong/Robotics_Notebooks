# OCS2 官方文档（leggedrobotics.github.io）

> 来源归档

- **标题：** OCS2 Documentation
- **类型：** site（官方 Sphinx 文档）
- **维护：** ETH Robotic Systems Lab (RSL) / leggedrobotics
- **链接：** https://leggedrobotics.github.io/ocs2/
- **对应仓库：** https://github.com/leggedrobotics/ocs2
- **入库日期：** 2026-09-24
- **一句话说明：** OCS2（Optimal Control for Switched Systems）C++ 工具箱的官方安装、建模、求解器与 ROS 部署文档；与 GitHub `main`（ROS 1）/`ros2` 分支对齐。

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| 文档 Overview | 明确 **BSD 3-Clause**；源码 **公开** 于 GitHub |
| GitHub README | **已开源**；ROS 2 见 `ros2` 分支与 `installation.md` |
| 示例 | `ocs2_robotic_examples/*` 含 double integrator、cartpole、ballbot、quadrotor、mobile manipulator、**legged robot** 等端到端 MPC |

## Overview 摘录（算法与定位）

OCS2 面向 **切换系统最优控制**，高效实现：

| 求解器 | 说明 |
|--------|------|
| **SLQ** | 连续时间域约束 DDP |
| **iLQR** | 离散时间域约束 DDP |
| **SQP** | 多重打靶 + **HPIPM** |
| **SLP** | 序列线性规划 + **PIPG** |
| **IPM** | 多重打靶非线性内点法 |

路径约束：**增广 Lagrangian** 或 **relaxed barrier**。机器人侧提供 URDF→动力学/代价/约束（含自碰撞、末端跟踪）、**CppAD** 自动微分与 **ROS** 接口，面向机载算力有限的实时 MPC。

## 文档导航（常用）

| 主题 | 典型入口 |
|------|----------|
| 总览 | `overview.html` |
| 安装 | Installation（ROS 1 文档站；ROS 2 见仓库 `ros2/installation.md`） |
| 入门 | Getting Started |
| 深度 | Doxygen / `ocs2_doc` 源码树 |

## 教程与引用（文档页列示）

- RSS 2021 MPC Workshop：Farbod Farshidian — OCS2 toolbox tutorial
- RSS 2021：Marco Hutter — Real-time optimal control for legged locomotion and manipulation
- 学术引用：`@misc{OCS2, ...}`（见 overview / README）

## 对 wiki 的映射

- [OCS2 实体页](../../wiki/entities/ocs2.md)
- [OCS2 仓库归档](../repos/ocs2.md)
- [Nonlinear MPC](../../wiki/methods/nonlinear-model-predictive-control.md)
- [MPC 求解器选型](../../wiki/queries/mpc-solver-selection.md)
