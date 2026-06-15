# MoveIt 官方门户（moveit.ai）

> 来源归档

- **标题：** MoveIt Motion Planning Framework（项目门户）
- **类型：** site（官方项目站）
- **维护：** MoveIt 社区；PickNik Robotics 主导开发
- **链接：** https://moveit.ai/
- **旧域名：** https://moveit.ros.org/（已重定向至 moveit.ai）
- **入库日期：** 2026-06-15
- **一句话说明：** MoveIt 开源运动规划与操作框架的 **官方门户**：能力概览、ROS 1/2 发行版矩阵、二进制/源码安装入口，并区分 **MoveIt 2（开源）** 与 **MoveIt Pro（商业）**。

## 为什么值得保留

- **一手版本矩阵**：门户列出 MoveIt 2 **Rolling / Jazzy LTS / Humble LTS** 与 MoveIt 1 **Noetic** 的维护状态，是选型 ROS 发行版与 MoveIt 分支的权威入口。
- **能力边界总览**：运动规划、操作/抓取、IK、控制接口、3D 感知、碰撞检测、RViz 交互、Gazebo 仿真、Setup Assistant、MoveIt Task Constructor（MTC）等模块以官方文案为准。
- 与本仓库 [MoveIt 2 实体页](../../wiki/entities/moveit2.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[ROS 2 官方文档](ros2-official-documentation.md) 形成「门户—文档—任务」闭环。

## 核心能力摘录（据 moveit.ai 首页）

| 模块 | 官方定位 |
|------|----------|
| **Motion Planning** | 高自由度轨迹生成，穿越 cluttered 环境 |
| **Manipulation** | 环境分析与抓取生成 |
| **Inverse Kinematics** | 给定末端位姿求关节解（含冗余臂） |
| **Control** | 时间参数化关节轨迹经通用接口下发底层控制器 |
| **3D Perception** | 深度传感器、点云与 Octomap 接入 |
| **Collision Checking** | 几何 primitive、mesh、点云避障 |
| **Setup Assistant** | 向导式配置任意机器人 MoveIt 包 |
| **Task Constructor** | 多子任务依赖的可组合 pick-and-place 管线 |

## 发行版矩阵（2026 前后，据门户）

| 发行名 | ROS | 状态 |
|--------|-----|------|
| **Rolling 2.13** | ROS 2 | 持续开发 |
| **Jazzy 2.12 LTS** | ROS 2 | **最新稳定，推荐** |
| **Humble 2.5 LTS** | ROS 2 | 维护中 |
| **Noetic 1.1 LTS** | ROS 1 | 维护中（MoveIt 1） |
| **MoveIt Pro** | ROS 2 | 商业支持（PickNik） |

## 配套一手入口

| 资源 | 链接 |
|------|------|
| **MoveIt 2 文档** | https://moveit.picknik.ai/ |
| **MoveIt 2 源码** | https://github.com/moveit/moveit2 |
| **MoveIt 1 源码** | https://github.com/moveit/moveit |
| **MoveIt 1 教程（Noetic）** | https://moveit.github.io/moveit_tutorials/ |
| **二进制安装（MoveIt 2）** | https://moveit.ros.org/install-moveit2/binary/ |
| **源码编译（MoveIt 2）** | https://moveit.ros.org/install-moveit2/source/ |
| **路线图** | https://moveit.ros.org/documentation/contributing/roadmap/ |
| **MoveIt Pro** | https://picknik.ai/pro |

## 对 wiki 的映射

- 升格页面：[wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)
- 交叉：[wiki/tasks/manipulation.md](../../wiki/tasks/manipulation.md)、[wiki/entities/curobo.md](../../wiki/entities/curobo.md)、[wiki/concepts/ros2-basics.md](../../wiki/concepts/ros2-basics.md)
