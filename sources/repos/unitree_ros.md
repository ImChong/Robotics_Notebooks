# unitree_ros

> 来源归档

- **标题：** unitree_ros
- **类型：** repo
- **来源：** Unitree Robotics（unitreerobotics）
- **链接：** https://github.com/unitreerobotics/unitree_ros
- **入库日期：** 2026-05-17
- **一句话说明：** 官方 ROS1 仿真与描述包集合：多机型 URDF/xacro、Gazebo8 世界与关节控制器（力矩/位置/角速度），README 明确 Gazebo 不提供高层行走；真机 ROS 控制需配合 `unitree_ros_to_real` 与 `unitree_legged_msgs`。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-ros.md`](../wiki/entities/unitree-ros.md)

## README 要点（编译自上游 master）

- **定位**：ROS 仿真包；可在 Gazebo 中加载机器人与关节控制器，做**底层**关节控制（力矩、位置、角速度）。Gazebo 仿真**不做高层控制（行走）**。
- **真机**：用 ROS 控制实机需 [`unitree_ros_to_real`](https://github.com/unitreerobotics/unitree_ros_to_real)；可用同一套 ROS 包做高低层控制（以该仓文档为准）。
- **机器人描述包（节选）**：`a1` / `a2` / `aliengo` / `b1` / `b2` / `b2w` / `dexterous_hand` / `g1` / `go1` / `go2` / `go2w` / `h1` / `h1_2` / `h2` / `laikago` / `r1` / `r1_air` / `z1` 等 `*_description`（完整列表见仓库 README）。
- **控制与仿真包**：`unitree_controller`、`z1_controller`、`unitree_gazebo`、`unitree_legged_control`（Gazebo 关节控制器，支持位置/速度/力矩模式；示例见 `unitree_controller/src/servo.cpp`）。
- **依赖**：ROS Melodic 或 Kinetic（README 写明 Kinetic 未充分测试）、Gazebo8；消息包 `unitree_legged_msgs` 来自 `unitree_ros_to_real` 仓库。
- **构建**：`catkin_make`；需按 README 修改 `unitree_gazebo/worlds/stairs.world` 中 `building_editor_models/stairs` 的本地路径。
- **Gazebo 启动示例**：`roslaunch unitree_gazebo normal.launch rname:=a1 wname:=stairs`（`rname` 在文档示例中含 laikago、aliengo、a1、go1 等；`wname` 可为 earth、space、stairs）。
- **Z1**：`roslaunch unitree_gazebo z1.launch`，控制侧指向 [z1 文档](https://dev-z1.unitree.com)。

## 与知识库其他条目的关系

| 本库条目 | 关系 |
|---------|------|
| [`unitree_ros_to_real.md`](unitree_ros_to_real.md) | 真机 ROS 桥、`unitree_legged_msgs` 来源；与仿真仓配套 |
| [`unitree_rl_mjlab.md`](unitree_rl_mjlab.md) | 当前官方 RL + MuJoCo 训练/部署主线；与 ROS1 Gazebo 栈并行存在 |
| [`wiki/entities/unitree-ros.md`](../../wiki/entities/unitree-ros.md) | 提炼页 |
