# autonomy_stack_go2

> 来源归档

- **标题：** Full Autonomy Stack for Unitree Go2
- **类型：** repo
- **来源：** Ji Zhang / CMU（jizhang-cmu）
- **链接：** https://github.com/jizhang-cmu/autonomy_stack_go2
- **星标（截至 2026-07-26）：** ~430+
- **主要语言：** C++
- **分类：** 四足自主导航 / SLAM / 规划
- **入库日期：** 2026-07-26
- **一句话说明：** 面向 Unitree Go2 EDU 的完整几何自主导航栈：内置 L1 LiDAR + 雷达 IMU，Point-LIO 建图，地形可通行性、避障、路径跟随与 FAR Planner。
- **沉淀到 wiki：** [autonomy-stack-go2](../../wiki/entities/autonomy-stack-go2.md)（实体）；[GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)、[point-lio-unilidar](../../wiki/entities/point-lio-unilidar.md)
- **相关：** [point_lio_unilidar](point_lio_unilidar.md)、[cmu-exploration.com](https://www.cmu-exploration.com)、[FAR Planner](https://github.com/MichaelFYang/far_planner)

---

## README 要点（编译自上游）

- **平台：** Go2 **EDU**（需 SDK）；可用机载或以太网外接算力。
- **传感器：** 仅用内置 **L1 LiDAR** 与雷达内 IMU（几何导航主线，非相机语义）。
- **模块：** SLAM（Point-LIO / `point_lio_unilidar` 适配）+ 地形可通行性分析 + 避障 + waypoint following + **FAR Planner** 路线规划。
- **模式：** 目标点自主导航；或手柄引导 + 系统负责避障。
- **依赖上游开源：** `unitreerobotics/point_lio_unilidar`、`unitree_ros2`、Unity `ROS-TCP-Endpoint` 等。

## 开源状态

- **已开源**：公开 GitHub 仓库（`jizhang-cmu/autonomy_stack_go2`）。
- **边界：** 解决的是 **几何定位与自主导航**，本身 **不是** SAM / 开放词汇语义建图。

## 对 wiki 的映射

- 实体：[autonomy-stack-go2](../../wiki/entities/autonomy-stack-go2.md)
- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)
- 几何基线：[point-lio-unilidar](../../wiki/entities/point-lio-unilidar.md)
- 导航总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
