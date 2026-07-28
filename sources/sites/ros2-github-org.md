# ROS 2 GitHub 组织（github.com/ros2）

> 来源归档

- **标题：** ROS 2（GitHub Organization）
- **类型：** site / org（上游源码组织入口）
- **来源：** Open Source Robotics Foundation / Open Robotics 与 ROS 2 社区
- **链接：** https://github.com/ros2
- **文档入口（org blog 字段）：** https://docs.ros.org/en/rolling
- **元仓库：** https://github.com/ros2/ros2（归档：[repos/ros2.md](../repos/ros2.md)）
- **公开仓数：** ~146（2026-07）
- **Followers：** ~4.6k
- **入库日期：** 2026-07-28
- **一句话说明：** ROS 2 官方源码组织：元仓库、客户端库、文档、示例、RMW 实现与工具链的上游索引；与 docs.ros.org 文档站互补。

## 为什么值得保留

- 已有 [ros2-official-documentation.md](ros2-official-documentation.md) 覆盖 **文档站**；本页覆盖 **源码组织拓扑**，避免「只会看 docs、找不到上游仓」的缺口。
- 选型与调试时常需定位 `rclcpp` / `rclpy` / `rmw_*` / `rosbag2` / `rviz` 等具体实现仓，组织页是导航根。
- 与本仓库 [ros2-basics](../../wiki/concepts/ros2-basics.md)、[tech-map ros2](../../tech-map/modules/system/ros2.md)、Nav2 / MoveIt / unitree_ros2 等实体形成「概念 → 上游 → 生态包」链。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 组织公开 | ✅ https://github.com/ros2 |
| 核心栈 | **已开源**（多仓；许可证因包而异，常见 Apache-2.0） |
| 二进制发行 | 官方推荐走发行版 apt / Docker，而非只 clone 本组织全部仓 |

## 高星仓索引（2026-07 API，节选）

| Stars | 仓库 | 角色 |
|------:|------|------|
| ~5.8k | [ros2/ros2](https://github.com/ros2/ros2) | **元仓库**：`ros2.repos` 拉取整树 |
| ~965 | [ros2/ros2_documentation](https://github.com/ros2/ros2_documentation) | docs.ros.org 源 |
| ~956 | [ros2/examples](https://github.com/ros2/examples) | 示例包 |
| ~780 | [ros2/rclcpp](https://github.com/ros2/rclcpp) | C++ 客户端库 |
| ~647 | [ros2/demos](https://github.com/ros2/demos) | 演示 |
| ~621 | [ros2/ros1_bridge](https://github.com/ros2/ros1_bridge) | ROS 1↔2 桥 |
| ~492 | [ros2/rmw_zenoh](https://github.com/ros2/rmw_zenoh) | Zenoh RMW |
| ~473 | [ros2/rviz](https://github.com/ros2/rviz) | 3D 可视化 |
| ~464 | [ros2/rclpy](https://github.com/ros2/rclpy) | Python 客户端库 |
| ~430 | [ros2/rosbag2](https://github.com/ros2/rosbag2) | 录制回放 |
| ~375 | [ros2/common_interfaces](https://github.com/ros2/common_interfaces) | 标准 msg/srv |
| ~261 | [ros2/ros2cli](https://github.com/ros2/ros2cli) | CLI 工具 |
| ~242 | [ros2/design](https://github.com/ros2/design) | 设计文档（design.ros2.org） |

> 导航、MoveIt、Autoware、Isaac ROS 等**上层栈**多在独立组织（`ros-navigation`、`moveit`、`autowarefoundation`、`NVIDIA-ISAAC-ROS`），不在本 org 内。

## 社区与开发入口（元仓 README 指向）

| 资源 | URL |
|------|-----|
| Discourse | https://discourse.ros.org/ |
| Zulip | https://openrobotics.zulipchat.com/ |
| Robotics Stack Exchange | https://robotics.stackexchange.com/ |
| ROSCon | https://roscon.ros.org |
| REP-2000（发行版与平台） | https://ros.org/reps/rep-2000.html |
| 学术引用 | DOI [10.1126/scirobotics.abm6074](https://www.science.org/doi/10.1126/scirobotics.abm6074) |

## 对 wiki 的映射

- [ros2-basics](../../wiki/concepts/ros2-basics.md)
- [dds-communication](../../wiki/concepts/dds-communication.md)
- [ros2-vs-lcm](../../wiki/comparisons/ros2-vs-lcm.md)
- [tech-map/modules/system/ros2.md](../../tech-map/modules/system/ros2.md)
- 元仓：[repos/ros2.md](../repos/ros2.md)
- 文档站：[ros2-official-documentation.md](ros2-official-documentation.md)
