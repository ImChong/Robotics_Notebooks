# ros2/ros2

> 来源归档

- **标题：** ros2/ros2（ROS 2 meta repository）
- **类型：** repo（元仓库 / vcs 清单）
- **来源：** Open Source Robotics Foundation / ROS 2 社区
- **链接：** https://github.com/ros2/ros2
- **组织：** https://github.com/ros2（归档：[sites/ros2-github-org.md](../sites/ros2-github-org.md)）
- **文档：** https://docs.ros.org（Humble 归档：[sites/ros2-official-documentation.md](../sites/ros2-official-documentation.md)）
- **Stars：** ~5.8k（2026-07）
- **Forks：** ~929
- **默认分支：** `rolling`
- **Homepage：** https://docs.ros.org
- **入库日期：** 2026-07-28
- **一句话说明：** ROS 2 **元仓库**：本身几乎不含应用代码，核心产物是 `ros2.repos`——用 vcstool 拉取 ament、rcl、rmw、DDS vendor、消息与工具等整棵工作区树。
- **沉淀到 wiki：** 是 → [`wiki/concepts/ros2-basics.md`](../../wiki/concepts/ros2-basics.md)

## 开源状态（2026-07-28）

**已开源**：元仓库 + `ros2.repos` 列出的上游仓均可公开 clone。日常用户应优先 **发行版二进制**（`apt install ros-<distro>-desktop` 等）；从源码构建整树面向贡献者与定制 RMW/发行版。

## README 定位（摘要）

> The Robot Operating System (ROS) is a set of software libraries and tools that help you build robot applications… And it's all open source.

- 入门：ROS.org getting-started → docs.ros.org Concepts / Beginner tutorials  
- 引用：Science Robotics DOI `10.1126/scirobotics.abm6074`  
- 发行版矩阵：REP-2000  

## 仓库内容结构

| 路径 / 文件 | 作用 |
|-------------|------|
| `README.md` | 社区、文档、开发资源索引 |
| `ros2.repos` | **核心**：`vcs import` 用的多仓清单（Rolling 分支约 **100+** `type: git` 条目） |
| `src/` | 本地工作区占位（由 vcstool 填入） |
| `pixi.toml` | 可选环境管理配置 |

## `ros2.repos` 拓扑（Rolling 抽样，2026-07）

| 族群 | 示例 URL | 角色 |
|------|----------|------|
| ament | `ament/ament_cmake` 等 | 构建与包索引 |
| RMW / DDS | `eProsima/Fast-DDS`、`eclipse-cyclonedds/cyclonedds` | 默认中间件实现 |
| 共享内存 | `eclipse-iceoryx/iceoryx` | 同机零拷贝路径相关 |
| 客户端库 | `ros2/rcl`、`ros2/rclcpp`、`ros2/rclpy`（清单内） | C/C++/Python API |
| 消息与接口 | `ros2/common_interfaces`、`ros2/unique_identifier_msgs` 等 | `.msg` / `.srv` / `.action` |
| 感知基础 | `ros-perception/image_common`、`laser_geometry` 等 | 图像/点云基础包 |
| 测试工具 | `osrf/osrf_testing_tools_cpp` | 测试基础设施 |

> 完整列表以当前分支 `ros2.repos` 为准；发行版分支（`humble`、`jazzy`、`kilted` 等）钉定对应版本。

## 典型源码工作区流程（官方习惯）

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
# 取得元仓中的 ros2.repos 后：
vcs import src < ros2.repos
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
```

生产部署仍推荐发行版包管理器；整树构建成本高、依赖多。

## 对 wiki 的映射

- [ROS 2 基础](../../wiki/concepts/ros2-basics.md) — 中间件定位与组件
- [DDS 通信](../../wiki/concepts/dds-communication.md) — RMW / Fast DDS / Cyclone
- [ROS 2 vs LCM](../../wiki/comparisons/ros2-vs-lcm.md)
- [tech-map ROS 2 模块](../../tech-map/modules/system/ros2.md)
- 组织索引：[ros2-github-org.md](../sites/ros2-github-org.md)
- 文档站：[ros2-official-documentation.md](../sites/ros2-official-documentation.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [lcm.md](lcm.md) | 底层高频运控对照 |
| [unitree_ros2.md](unitree_ros2.md) | 厂商 ROS 2 包（消费 CycloneDDS） |
| [navigation2.md](navigation2.md) / [moveit-moveit2.md](moveit-moveit2.md) | 上层栈，独立组织维护 |
| [plotjuggler.md](plotjuggler.md) | rosbag2 / topic 调试 |
