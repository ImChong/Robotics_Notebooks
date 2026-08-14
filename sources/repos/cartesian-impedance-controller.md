# Cartesian Impedance Controller（Mayr / Lund）

> 来源归档

- **标题：** Cartesian Impedance Controller
- **类型：** repo
- **来源：** Matthias Mayr / Julian M. Salt-Ducaju（隆德大学）
- **链接：** <https://github.com/matthias-mayr/Cartesian-Impedance-Controller>
- **项目页：** <https://matthias-mayr.github.io/Cartesian-Impedance-Controller/>
- **论文：** JOSS <https://doi.org/10.21105/joss.05194>；arXiv <https://arxiv.org/abs/2212.11215>
- **许可：** BSD-3-Clause
- **入库日期：** 2026-08-13
- **一句话说明：** 力矩控制机械臂的笛卡尔阻抗 C++ 库 + ros2_control 插件；主任务柔顺，次级任务经雅可比零空间做关节阻抗（7 轴 Panda/FR3、iiwa7）。
- **沉淀到 wiki：** [`wiki/entities/paper-cartesian-impedance-controller.md`](../../wiki/entities/paper-cartesian-impedance-controller.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | 非 fork；language C++；约 342 stars；默认分支 `master`（ROS 2 CI 跟 `ros2` 工作流） |
| 许可 | BSD-3-Clause |
| 可运行入口 | `scripts/install_dependencies.sh` → `colcon build`；Docker；`test/base_tests` 不依赖仿真 |
| 关键源文件 | `src/cartesian_impedance_controller.cpp`（基库力矩叠加）、`src/cartesian_impedance_controller_ros.cpp`（ROS 插件）、`src/pseudo_inversion.h` |
| 真机 | FR3 + `franka_ros`；iiwa7 + `lbr_fri_ros2_stack` / 历史 `iiwa_ros` |

**结论：已开源。**

## 仓库入口

| 组件 | 说明 |
|------|------|
| 基库 | 少依赖；可嵌入 DART 或任意能下发关节力矩的仿真 |
| ROS 2 | `cartesian_impedance_controller/CartesianImpedanceController`；YAML 配 7 关节 + `nullspace_stiffness` |
| 运行时话题 | `/set_cartesian_stiffness`、`/set_damping_factors`、`/set_cartesian_wrench`、`/follow_joint_trajectory` |
| 调参注意 | MoveIt 关节轨迹需要 **非零零空间刚度**，否则只跟末端、丢掉肘部构型 |
| 依赖 | Eigen；ROS 侧用 RBDyn / SpaceVecAlg 算 FK 与 Jacobian |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-cartesian-impedance-controller](../../wiki/entities/paper-cartesian-impedance-controller.md) | JOSS 论文实体与时序图 |
| [null-space-control](../../wiki/concepts/null-space-control.md) | 投影公式与 7 轴选型 |
| [libfranka](./libfranka.md) | 厂商单机示例；本仓补多机型与在线零空间 |
| [franka-research-3](../../wiki/entities/franka-research-3.md) | 已部署的 7 轴科研臂 |
