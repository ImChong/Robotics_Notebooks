# unitree_ros_to_real

> 来源归档

- **标题：** unitree_ros_to_real
- **类型：** repo
- **来源：** Unitree Robotics（unitreerobotics）
- **链接：** https://github.com/unitreerobotics/unitree_ros_to_real
- **入库日期：** 2026-05-17
- **一句话说明：** 从 ROS 向实机下发控制的配套包：`unitree_legged_msgs` 与 `unitree_legged_real`；README 推荐 Ubuntu 18.04 + ROS Melodic，并与指定版本的 `unitree_legged_sdk`（如 v3.8.0 面向 Go1）同工作空间编译。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-ros.md`](../wiki/entities/unitree-ros.md)（与 `unitree_ros` 同页归纳）

## README 要点（编译自上游 master）

- **功能**：经 ROS 向真机发送控制指令；支持**低层**（全关节）与**高层**（行走方向/速度）控制。
- **版本语境**：文档示例为 Packages v3.8.0，适配 `unitree_legged_sdk` v3.5.1、**Go1**；新版本 SDK 发行说明需对照具体机型（如 v3.8.0 仅 Go1；A1 需查历史 tag）。
- **包**：`unitree_legged_msgs`（消息）、`unitree_legged_real`（ROS↔真机接口）。
- **网络**：PC 与机器人网线连接后配置 `ipconfig.sh` / `interfaces` 静态地址（示例网段 192.168.123.x）。
- **运行**：`roslaunch unitree_legged_real real.launch ctrl_level:=highlevel|lowlevel`；示例节点 `ros_example_walk`、`ros_example_postion`（上游拼写）、`state_sub`、`keyboard_control.launch`。
- **低层控制安全提示**：README 要求特定手柄组合进入关节控制模式并**吊起机器人**后再跑低层例程。

## 与 `unitree_ros` 的关系

[`unitree_ros`](unitree_ros.md) 的依赖说明中要求将本仓的 `unitree_legged_msgs` 一并放入 catkin 工作空间，形成「仿真描述 + Gazebo 控制」与「真机 ROS 桥」的分层组合。
