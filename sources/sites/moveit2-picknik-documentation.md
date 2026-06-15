# MoveIt 2 官方文档（PickNik / moveit.picknik.ai）

> 来源归档

- **标题：** MoveIt 2 Documentation（统一教程、概念、How-To、API）
- **类型：** site（官方文档）
- **维护：** MoveIt 社区；文档托管于 PickNik 域
- **链接：** https://moveit.picknik.ai/main/（Rolling）；按 ROS 发行版另有 `humble/`、`jazzy/` 等分支文档
- **教程源码仓：** https://github.com/moveit/moveit2_tutorials
- **入库日期：** 2026-06-15
- **一句话说明：** MoveIt 2 的 **权威技术文档**：从 Quickstart、Setup Assistant 到 **move_group**、**Planning Scene Monitor**、规划插件（OMPL / Pilz / CHOMP）、**Hybrid Planning**、**MoveIt Task Constructor** 与 ros2_control 集成。

## 为什么值得保留

- **架构概念一手定义**：`move_group` 节点、Motion Planning Plugin 接口、Planning Scene、适配器链（边界检查、碰撞、时间参数化）均以本文档为准。
- **与 MoveIt 1 文档分离**：MoveIt 1 教程仍见 [moveit.github.io/moveit_tutorials](https://moveit.github.io/moveit_tutorials/)；MoveIt 2 以 picknik.ai 为统一入口（门户自述 2019 首发 ROS 2 版，2022 doc-a-thon 大更新）。
- 机器人集成需对照 **URDF + SRDF + moveit_config** 三件套；SRDF 解析见 [ros-planning/srdfdom](../repos/ros-planning-srdfdom.md)。

## 文档结构摘录

| 分区 | 内容 |
|------|------|
| **Tutorials** | 安装、Quickstart、Setup Assistant、Pick and Place（含 MTC） |
| **Examples** | MoveGroup C++ Interface、Planning Scene / PSM、ROS API 教程 |
| **Concepts** | Kinematics、Motion Planning、Hybrid Planning、move_group、Planning Scene Monitor、Trajectory Processing、MTC |
| **How-To Guides** | 迁移、调试、API Doxygen 本地生成等 |
| **API** | C++ API 参考 |

## 核心概念摘录（Concepts）

### Motion Planning（插件化）

- MoveIt 通过 **plugin interface** 对接不同规划库；默认经 **Setup Assistant** 配置 **OMPL**；亦内置 **Pilz industrial motion planner**、**CHOMP**。
- 用户经 `move_group` 的 **ROS action/service** 提交 **MotionPlanRequest**（关节空间目标或末端位姿、默认碰撞检查含自碰与附着物体），得到 **MotionPlanResult**。
- **Planning adapters** 链示例：`CheckStartStateBounds`、`ValidateWorkspaceBounds`、`CheckStartStateCollision`、`AddTimeParameterization`、`ResolveConstraintFrames`。

### move_group 节点

- MoveIt 的 **关键 ROS 节点**：聚合运动规划、场景监控、执行接口；用户通过其 action/service 访问规划能力。
- 依赖 **URDF**（运动学/几何）、**SRDF**（语义：规划组、碰撞豁免、末端执行器）与 **MoveIt configuration**（规划器、控制器映射等）。

### Planning Scene Monitor

- 维护 **Planning Scene**（机器人状态 + 世界几何 + 碰撞对象）。
- **World geometry monitor** 融合机器人传感器（LiDAR/深度相机）与用户输入；3D 感知经 **occupancy map monitor**（可插拔 updater，含 depth image updater）。
- 与 Octomap 等占用栅格表示衔接。

### MoveIt Task Constructor（MTC）

- 以 **可组合 stage** 定义 pick-and-place 等多子任务依赖管线（官方 Concepts 独立章节 + Pick and Place 教程）。

## 对 wiki 的映射

- 升格页面：[wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)
- 原始门户：[moveit-official-portal.md](moveit-official-portal.md)
- 代码仓：[moveit-moveit2.md](../repos/moveit-moveit2.md)

## 参考链接

- MoveIt 2 文档首页：https://moveit.picknik.ai/main/
- Motion Planning 概念：https://moveit.picknik.ai/main/doc/concepts/motion_planning.html
- move_group 概念：https://moveit.picknik.ai/main/doc/concepts/move_group.html
- Planning Scene Monitor：https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html
- moveit2_tutorials：https://github.com/moveit/moveit2_tutorials
