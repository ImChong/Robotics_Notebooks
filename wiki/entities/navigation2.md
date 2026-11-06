---
type: entity
tags: [repo, ros2, navigation, nav2, mobile-robot, planning, linux-foundation]
status: complete
updated: 2026-09-20
related:
  - ./python-robotics.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../concepts/ros2-basics.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ./slam-toolbox.md
  - ./cartographer.md
  - ./fast-lio.md
  - ./mushr.md
  - ./oomwoo.md
  - ./autoware.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ./awesome-physical-ai-natnew.md
original:
  - ../../sources/papers/navigation2_marathon2_reading_arxiv_2003_00368.md
sources:
  - ../../sources/repos/navigation2.md
  - ../../sources/repos/awesome-physical-ai-union-catalog.md
  - ../../sources/repos/awesome-physical-ai-natnew.md
summary: "Navigation2（Nav2）是 ROS 2 标准导航框架：行为树、全局/局部规划插件、代价地图与恢复行为，承接 SLAM 输出的 map/odom 并输出 cmd_vel。"
---

# Navigation2（Nav2）

**Navigation2**（[ros-navigation/navigation2](https://github.com/ros-navigation/navigation2)）是 ROS 2 生态的 **移动机器人导航参考实现**，将定位结果、静态/动态代价地图与路径跟踪解耦为可替换插件。

## 一句话定义

Nav2 是以行为树编排规划、控制、恢复等任务服务器，并以运行时插件和生命周期节点组织的 ROS 2 移动机器人导航框架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| LiDAR | Light Detection and Ranging | 激光雷达，地形感知与建图主传感器 |
| LIO | LiDAR-Inertial Odometry | 激光-惯性里程计 |
| VIO | Visual-Inertial Odometry | 视觉-惯性里程计，融合相机与 IMU 估计位姿 |

## 为什么重要

- **事实标准**：AMR、服务机器人、科研平台普遍以 Nav2 为 **规划—控制中枢**。
- **插件化**：全局规划（NavFn、Smac Planner 等）、局部规划（DWB、RPP、MPPI 等）、控制器与 **行为树（BT）** 可独立升级。
- 与 [ROS 2 基础](../concepts/ros2-basics.md)、[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) 形成常见 **「建图 + 导航」** 闭环。
- 算法预习可先用 [PythonRobotics](./python-robotics.md) 跑通 A*、DWA、Stanley/MPC 等示例，再读 Nav2 插件与 ROS 2 集成。
- 开源家用清扫整机入口见 [OOMWOO](./oomwoo.md)（Gazebo + slam_toolbox + Nav2，仿真优先）。

## 核心结构/机制

| 组件 | 说明 |
|------|------|
| **bt_navigator** | 行为树编排：导航、充电、恢复（spin、backup、wait） |
| **planner_server** | 全局路径；支持 2D grid、Hybrid-A* 等 |
| **controller_server** | 局部跟踪；输出 `geometry_msgs/Twist` |
| **smoother_server** | 路径平滑 |
| **costmap_2d** | 静态层 + 膨胀层 + 可选 3D 源（如 nvblox） |
| **lifecycle** | 节点生命周期管理，便于量产启动顺序 |

典型数据流：**SLAM/AMCL** 提供 `map`↔`odom`↔`base_link` → **global costmap** → **planner** → **local costmap** → **controller** → 底盘驱动。

## 核心机制（原论文视角）

依据 [The Marathon 2: A Navigation System](https://arxiv.org/abs/2003.00368)（已通读，见 [归档](../../sources/papers/navigation2_marathon2_reading_arxiv_2003_00368.md)）：

- **两种设计模式**：行为树导航器（XML，运行时加载，基于 BehaviorTree.CPP）+ 任务专用异步服务器（planner / controller / recovery，各自承载环境模型与插件），BT 节点通过 ROS 2 action 调用服务器，可跨进程、跨 CPU 运行。
- **生命周期节点**：所有服务器为 Managed Node，由 Lifecycle Manager 驱动 configure→activate，保证确定性启动与关闭。
- **恢复行为**：按「保守到激进」排序（清代价地图 → 原地旋转 → 等待）；全局规划 1 Hz，规划失败时才逐级升级。
- **分层代价地图**：Static / Inflation / Voxel / STVL 等层可叠加，STVL 维护带时间衰减的稀疏 3D 体素。

## 工程实践

- **长时验证数据（论文）**：两台差速机器人（Tiago、RB-1）在校园限速 0.45 m/s，共 22.8 小时、37.4 英里，168 次恢复（4.3 次/英里），0 碰撞、0 急停。
- **恢复触发的主因**：拥挤时无法规划；长走廊重复特征 + 人流导致 AMCL 置信度下降（Spin 有助重定位）。恢复次数多不等于性能差，它是容错机制的一部分。
- **CI**：Gazebo 仿真集成测试进入持续集成，用于回归与算法性能比较。
- 论文发表于 2020 年，其中 NavFn+TEB 的实验配置不等于当前默认插件（Smac、MPPI、RPP 等不在该文范围）；当前行为以官方文档与源码为准。

## 常见误区或局限

- **误区：装好 Nav2 即可自动驾驶** — 仍需可靠定位、地图质量、传感器标定与安全监控。
- **误区：默认 DWB 适用所有底盘** — 全向、阿克曼、差速动力学差异大，需换 **controller / footprint**。
- **局限（论文）**：论文所用 A* 规划器对非圆形非完整约束机器人（如车型）不给出可行路径；实验未使用显式动态障碍检测与预测。
- **局限**：Nav2 面向 **平面移动底盘**；人形/腿式 **步态规划** 不在其范围（见 [OpenLoong](./openloong.md)）。

## 源码运行时序图

入口：`ros2 launch nav2_bringup bringup_launch.py`（launch 文件 `nav2_bringup/launch/bringup_launch.py`）。下图为 launch 层的启动顺序，以及生命周期管理器对各节点的 configure 与 activate；`bringup_launch.py` 中的 `use_composition` 为真时，各服务改为同一容器内的 ComposableNode，图中未展开。

```mermaid
sequenceDiagram
  participant User as ros2 launch 命令行
  participant BU as bringup_launch.py
  participant LOC as localization_launch.py
  participant NAV as navigation_launch.py
  participant Srv as 各 server 节点 controller planner behavior 等
  participant BT as bt_navigator main.cpp 与 BtNavigator
  participant LCM as nav2_lifecycle_manager lifecycle_manager
  User->>BU: 启动 bringup_launch.py，声明参数
  alt slam 且 use_localization 为真
    BU->>BU: 包含 slam_launch.py
  else 其余情况
    BU->>LOC: IncludeLaunchDescription 包含 localization_launch.py
    LOC-->>BU: map_server 与 amcl 节点
  end
  BU->>NAV: IncludeLaunchDescription 包含 navigation_launch.py
  NAV->>Srv: Node 启动 controller_server、smoother_server、planner_server、route_server、behavior_server
  NAV->>BT: Node 启动 bt_navigator，main 中 rclcpp::spin
  NAV->>Srv: Node 启动 waypoint_follower、velocity_smoother、collision_monitor、docking_server、following_server
  NAV->>LCM: Node 启动 lifecycle_manager_nav2，传入 autostart 参数与节点名列表
  LCM->>Srv: 依次 configure 各节点
  LCM->>BT: configure，BtNavigator::on_configure 加载默认导航器 navigate_to_pose 与 navigate_through_poses
  alt autostart 为真
    LCM->>Srv: activate 各节点
    LCM->>BT: on_activate 激活导航器
  end
```

图依据：ros-navigation/navigation2@20b9fa3，`nav2_bringup/launch/bringup_launch.py`、`navigation_launch.py`、`localization_launch.py`、`nav2_bt_navigator/src/main.cpp`、`nav2_bt_navigator/src/bt_navigator.cpp`（on_configure、on_activate 段）；未运行代码。

## 参考来源

- [sources/repos/navigation2.md](../../sources/repos/navigation2.md)
- [sources/repos/navigation_slam_autonomy_stack_catalog.md](../../sources/repos/navigation_slam_autonomy_stack_catalog.md)
- [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2)

- 源码核对（2026-11-02）：ros-navigation/navigation2@20b9fa3，读过 `README.md`、`nav2_bringup/launch/bringup_launch.py`、`navigation_launch.py`、`localization_launch.py`、`nav2_bt_navigator/src/main.cpp`、`nav2_bt_navigator/src/bt_navigator.cpp`（部分）；时序图依据以上文件，未运行代码。

## 关联页面

- [PythonRobotics](./python-robotics.md) — 规划/跟踪算法入门代码
- [导航·SLAM·自动驾驶栈总览](../overview/navigation-slam-autonomy-stack.md)
- [LiDAR / LIO / VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)
- [OOMWOO](./oomwoo.md) — 开源家用扫地机（Nav2 应用整机）
- [Language-to-Navigation-Goals（VLM+RGB-D→Nav2）](./paper-language-to-navigation-goals-rgbd.md)
- [Autoware](./autoware.md)

## 推荐继续阅读

- [PythonRobotics 在线教材](https://atsushisakai.github.io/PythonRobotics/)（算法预习）
- [Nav2 官方文档](https://docs.nav2.org/)
