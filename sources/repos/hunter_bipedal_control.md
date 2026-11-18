# hunter_bipedal_control

> 来源归档（国内具身开源全景）

- **标题：** hunter_bipedal_control
- **类型：** repo
- **机构：** 桥介数物
- **链接：** https://github.com/bridgedp/hunter_bipedal_control
- **分类：** 运动控制
- **入库日期：** 2026-09-06（2026-11-18 依据官方 README 与源码深化）
- **一句话说明：** EC-Hunter80-V01 双足机器人使用的 ROS1 Noetic 开源运控框架，组合 OCS2 SQP-MPC、状态估计和基于 qpOASES 的 WeightedWbc；支持 Gazebo、MuJoCo 和真机入口。
- **项目主页：** https://bridgedp.github.io/hunter_bipedal_control/
- **许可证：** MIT（仓库 LICENSE）
- **源码版本：** `main@37310dde100e2e8373fc7c2c02e825c358e6fd2e`
- **核对范围：** 已通读仓库 README.md（安装、仿真、真机、无手柄操作、问题提醒、参考文献），阅读关键源码 `legged_controllers/src/LeggedController.cpp`、`TargetTrajectoriesPublisher.cpp` 与 `.h`、`legged_wbc/src/WeightedWbc.cpp`、`one_start*.launch`，并核对仓库目录结构；未编译或上机运行。
- **通读：** 已完成
- **官方项目页归档：** [hunter-bipedal-control](../sites/hunter-bipedal-control.md)
- **沉淀到 wiki：** [`wiki/entities/cn-os-hunter-bipedal-control.md`](../../wiki/entities/cn-os-hunter-bipedal-control.md)

## 开源状态

- **已开源**：公开仓库提供控制器、状态估计、WBC、ROS launch、Gazebo/MuJoCo 适配及桥接硬件接口源码；本次核对的是源码可获取性，不等于复现运行通过。
- 官方项目页明确提供 [Code](https://github.com/bridgedp/hunter_bipedal_control) 入口；不将本仓库的 MPC/WBC 算法误写成 RL/VLA 训练代码。
- 依赖 OCS2、ROS1 Noetic；仅 MuJoCo 路径额外要求 LCM。真机启动依赖特定硬件/总线，未验证通用适配。

## 可核验的代码与运行入口

| 部分 | 代码 / 入口 | 职责 |
|---|---|---|
| 控制器 | `legged_controllers/src/LeggedController.cpp` | `SqpMpc` / `MPC_MRT_Interface`、Kalman 状态估计、`WeightedWbc`、HybridJointInterface |
| 目标轨迹 | `legged_controllers/src/TargetTrajectoriesPublisher.cpp`、`include/legged_controllers/TargetTrajectoriesPublisher.h` | 订阅 `/cmd_vel` 并使用 OCS2 `TargetTrajectoriesRosPublisher` 发送参考轨迹 |
| WBC | `legged_wbc/src/WeightedWbc.cpp` | 浮动基动力学/力矩限幅/摩擦锥约束 + 加权任务，经 qpOASES 求 QP |
| 仿真 | `legged_controllers/launch/one_start_gazebo.launch`、`one_start_mujoco.launch` | 对接 Gazebo 或 MuJoCo 的 ROS 控制接口 |
| 真机 | `legged_controllers/launch/one_start_real.launch` | 对接 `legged_bridge_hw` |

## 对 wiki 的映射

- [wiki/entities/cn-os-hunter-bipedal-control.md](../../wiki/entities/cn-os-hunter-bipedal-control.md)
