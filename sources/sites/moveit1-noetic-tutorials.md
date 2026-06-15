# MoveIt 1 官方教程（Noetic / moveit_tutorials）

> 来源归档

- **标题：** MoveIt Tutorials — moveit_tutorials Noetic documentation
- **类型：** site（官方教程，ROS 1）
- **链接：** https://moveit.github.io/moveit_tutorials/
- **源码仓：** https://github.com/moveit/moveit_tutorials（ROS 1 分支/版本）
- **入库日期：** 2026-06-15
- **一句话说明：** **MoveIt 1（ROS 1）** 的经典教程站点；文档内版本下拉可切换 **MoveIt 2 Humble/Rolling** 与 **MoveIt 1 Melodic/Noetic**，是理解 MoveIt 概念在 ROS 1 时代落地方式的 **一手入口**。

## 为什么值得保留

- **历史与迁移对照**：许多实验室/工业现场仍运行 **Noetic + MoveIt 1.1**；概念（Planning Scene、move_group、Setup Assistant、OMPL）与 MoveIt 2 同源，但 API 与 launch 体系不同。
- **版本切换锚点**：教程页 header 直接链接 `moveit.picknik.ai/humble` 与 `main`，便于对照 ROS 1→2 迁移（亦见 moveit2 仓 `doc/MIGRATION_GUIDE.md`）。

## 教程结构（Noetic 版，概览）

| 区块 | 典型主题 |
|------|----------|
| **Getting Started** | 安装、Quickstart、RViz 插件 |
| **Concepts** | Robot Model、Planning Scene、Kinematics、Motion Planning |
| **Examples** | MoveGroup Interface、Planning Scene API、Pick and Place |
| **Configuration** | Setup Assistant、Controllers、Perception |

## 与 MoveIt 2 的关系

- **MoveIt 1 代码仓：** [moveit/moveit](https://github.com/moveit/moveit)（ROS 1）
- **MoveIt 2 文档：** [moveit2-picknik-documentation.md](moveit2-picknik-documentation.md)
- 新 ROS 2 项目应优先 **Jazzy/Humble + MoveIt 2**；Noetic 维护窗口与 ROS 1 EOL 策略须以 OSRF/社区公告为准。

## 对 wiki 的映射

- 交叉页面：[wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)（含 MoveIt 1 谱系小节）
- 代码仓：[moveit-moveit1.md](../repos/moveit-moveit1.md)
