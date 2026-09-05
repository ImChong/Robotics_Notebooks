# zenoh_ros2_sdk

> 来源归档

- **标题：** zenoh-ros2-sdk — Python ROS 2 over Zenoh without ROS 2 install
- **类型：** repo
- **组织：** [ROBOTIS-GIT](https://github.com/ROBOTIS-GIT)
- **链接：** <https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk>
- **文档：** <https://robotis-git.github.io/zenoh_ros2_sdk/>
- **Stars：** ~27（2026-09-05）
- **入库日期：** 2026-09-05
- **一句话说明：** 纯 Python 经 Zenoh 发布/订阅 ROS 2 话题与服务，无需本机 ROS 2 安装；话题出现在 `ros2 topic list`（liveliness discovery），自动加载 msg/srv 定义与 type hash；与 `lerobot_robot_ros2_zenoh` 配套供 LeRobot 采训推。

## 关联仓

| 仓 | 角色 |
|----|------|
| [lerobot_robot_ros2_zenoh](https://github.com/ROBOTIS-GIT/lerobot_robot_ros2_zenoh) | LeRobot `ROS2ZenohConfig` 机器人接口（α）；读 `/joint_states`、发 trajectory、相机 topic |
| [physical_ai_tools](physical_ai_tools.md) | 真机 LeRobot + ROS 2 主界面（传统 ROS 2 路径） |

## 开源状态

**已开源**（pip 包 `zenoh-ros2-sdk`，CLI `zenoh-ros2`）；`lerobot_robot_ros2_zenoh` 标 **Alpha**，需本地 editable 安装 `zenoh_ros2_sdk` 与 LeRobot 插件发现。

## 对 wiki 的映射

- 组织 hub：[robotis.md](../../wiki/entities/robotis.md)
- LeRobot 交叉：[lerobot.md](../../wiki/entities/lerobot.md)、[robotis-physical-ai-tools.md](../../wiki/entities/robotis-physical-ai-tools.md)
