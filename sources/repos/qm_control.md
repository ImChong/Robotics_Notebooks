# qm_control（四足机械臂 OCS2 MPC + WBC）

- **标题：** qm_control
- **类型：** repo
- **仓库：** <https://github.com/skywoodsz/qm_control>
- **许可：** BSD-3-Clause
- **栈：** ROS1 Noetic · Gazebo · **OCS2** NMPC · **WBC** · catkin
- **收录日期：** 2026-09-24
- **开源结论：** **已开源**（~349★；`main` + `feature-force` / `feature-compliance` / `feature-real` 分支；README 标明仍在开发）

## 一句话摘要

面向 **四足机械臂（quadruped manipulator）** 的 **MPC + 全身控制** 参考实现：基于 **OCS2** 做全身/末端规划，Gazebo 仿真，支持 **仅 MPC** 与 **MPC-WBC** 两版；分支分别覆盖 **力扰动稳定**、**全身柔顺** 与 **真机**。

## 为何值得保留

- **OCS2 下游实例：** 与 [OCS2](../../wiki/entities/ocs2.md)、[Centroidal NMPC + WBC 栈](../../wiki/methods/centroidal-nmpc-wbc-stack.md) 知识链对齐，展示 **腿足基座 + 操作臂** 的 loco-manipulation 模型控制管线（非 RL）。
- **工程可跑：** catkin 构建、`qm_gazebo` + `qm_controllers` launch；手柄分控基座与末端；README 给出末端在基座行走 30 cm 时 **≤3.5 mm / 2.6°** 稳定误差仿真图。
- **论文背书：** README 链 IROS 2024 录用文（whole-body compliance、关节力矩与地面摩擦饱和）及 HIT 学位论文（视觉伺服 + 集值反馈动态抓取，中国境内）。

## 技术要点（编译自 README）

| 项 | 内容 |
|----|------|
| 依赖 | [OCS2](https://leggedrobotics.github.io/ocs2/installation.html) · ROS Noetic |
| 包结构 | `qm_gazebo` · `qm_controllers` · `qm_wbc` · `qm_estimation` · `qm_interface` · `qm_msgs` · `qm_description` |
| 模式 | **MPC-WBC**（`empty_world.launch` + `load_controller.launch`）vs **MPC only**（`*_mpc.launch`） |
| 分支 | `main`：全身运动、假设末端无外力；`feature-force`：末端力扰动稳定；`feature-compliance`：全身柔顺；`feature-real`：硬件 |
| 控制入口 | `rqt_controller_manager` 启控；`load_qm_target.launch` / rviz 发命令 |

## 对 Wiki 的映射

- [qm-control 实体页](../../wiki/entities/qm-control.md)
- 交叉 [legbot-mpc-wbc](../../wiki/entities/legbot-mpc-wbc.md)（四足 MPC–WBC 对照）、[OCS2](../../wiki/entities/ocs2.md)

## 参考来源（原始）

- 代码：<https://github.com/skywoodsz/qm_control>
- 视频：[YouTube](https://youtu.be/JCn5obOh4D8) · [Bilibili](https://www.bilibili.com/video/BV1uP411v7Ab)
- 论文：Zhang et al., IROS 2024（whole-body compliance，README 引用）
