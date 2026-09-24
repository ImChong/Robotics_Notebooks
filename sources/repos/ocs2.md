# OCS2

> 来源归档

- **标题：** OCS2 Toolbox
- **类型：** repo / mpc-framework
- **维护：** ETH Robotic Systems Lab (RSL) — [leggedrobotics/ocs2](https://github.com/leggedrobotics/ocs2)
- **链接：** https://github.com/leggedrobotics/ocs2
- **文档：** https://leggedrobotics.github.io/ocs2/
- **许可：** BSD 3-Clause（`LICENCE.txt`）
- **入库日期：** 2026-07-30（2026-09-24 按官方 README + 文档站深化）
- **一句话说明：** C++ **切换系统最优控制** 工具箱，强调机器人 **实时 NMPC**；多求解器（DDP/iLQR/SQP/SLP/IPM）、URDF/Pinocchio 建模、CppAD 求导与 ROS 1/2 示例（含腿式与移动操作）。
- **开源状态：** **已开源**（完整源码 + 文档 +  robotic examples）
- **沉淀到 wiki：** [OCS2](../../wiki/entities/ocs2.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub README | **已开源**；`main` = ROS 1，`ros2` 分支 + `installation.md` = ROS 2 |
| 官方文档站 | 与仓库一致；Overview 列算法、许可、引用与教程链接 |
| 可运行入口 | `ocs2_robotic_examples/*` + ROS nodes；`ocs2_python_interface` 部分 Python 绑定；`ocs2_mpcnet` 学习策略相关 |

## 仓库结构（README 摘要）

| 包 / 目录 | 角色 |
|-----------|------|
| `ocs2_core` | 核心类型、rollout、AD 工具 |
| `ocs2_oc` | OCP 构建块（代价、约束、动力学、参考管理） |
| `ocs2_ddp`, `ocs2_mpc` | DDP 族 + MPC 运行时接口 |
| `ocs2_sqp`, `ocs2_slp`, `ocs2_ipm` | SQP / SLP / IPM 求解器 |
| `ocs2_pinocchio/*` | URDF、质心模型、自碰撞（HPP-FCL）、可视化 |
| `ocs2_ros_interfaces`, `ocs2_msgs` | ROS 消息与节点 |
| `ocs2_robotic_examples/*` | 双积分器、倒立摆、ballbot、四旋翼、移动操作、**腿式** 等 |
| `ocs2_mpcnet` | MPC-Net 训练/部署工具 |
| `ocs2_doc` | Sphinx / Doxygen 文档源 |

## 摘录要点（相对主表策展的增量）

- **切换系统：** mode schedule、jump map，支持单域/多域 OCP（接触切换、步态相位）。
- **约束处理：** 硬/软约束 + 增广 Lagrangian / relaxed barrier。
- **求导：** 解析接口 + **CppAD** / 可选 **CppADCodeGen**。
- **生态：** 与 [Pinocchio](../../wiki/entities/pinocchio.md)、**HPIPM**（SQP 后端）同栈；ETH ANYmal / 四足 NMPC 论文链路的常用开源实现入口。

## 为什么值得保留

- 人形/腿足 **模型预测控制 + WBC** 选型时的 **一等公民**（与 acados、crocoddyl 对照）。
- 文档与示例完整，适合从 URDF 到 ROS MPC 节点走通 **System 1** 控制栈。
- 大量 RSL 论文（legged locomotion、whole-body MPC、MPC-Net 等）以 OCS2 为复现/扩展基座。

## 对 wiki 的映射

- [OCS2](../../wiki/entities/ocs2.md)
- [OCS2 官方文档归档](../sites/ocs2-official-docs.md)
- [Nonlinear MPC](../../wiki/methods/nonlinear-model-predictive-control.md)
- [Centroidal NMPC + WBC stack](../../wiki/methods/centroidal-nmpc-wbc-stack.md)
- [MPC solver selection](../../wiki/queries/mpc-solver-selection.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)（策展主表入口，历史）
