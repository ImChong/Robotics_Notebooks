# FAST-LIO

> 来源归档

- **标题：** FAST-LIO
- **类型：** repo
- **链接：** https://github.com/hku-mars/FAST_LIO
- **原文：** https://github.com/hku-mars/FAST_LIO/blob/7cc4175/README.md
- **核对版本：** `7cc4175`（官方仓库源码快照）
- **复核日期：** 2026-11-17
- **核对范围：** 已通读官方 `README.md` 全文（Related Works、FAST-LIO 与 FAST-LIO2、Prerequisites、Build、Directly run、Rosbag Example、UAV、Acknowledgments）；另核对同版本 `launch/mapping_mid360.launch`、`config/mid360.yaml`、`CMakeLists.txt`，以及 `src/laserMapping.cpp`、`src/preprocess.cpp`、`src/IMU_Processing.hpp` 的主要回调、处理入口、主循环与建图发布段。没有逐行审计完整源码、滤波器子模块或复现全部实验。
- **通读：** 已完成
- **代码：** https://github.com/hku-mars/FAST_LIO
- **开源状态：** 已开源（ROS 1 C++/catkin 及相关源码，含依赖子模块）
- **Stars：** ~4709（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 紧耦合 LiDAR-惯性里程计：ikd-Tree、迭代 ESKF，高效鲁棒。
- **沉淀到 wiki：** [lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

---

## 核心定位

FAST-LIO / FAST-LIO2 是以 IMU 传播和紧耦合迭代卡尔曼更新为核心的 LiDAR-inertial odometry 项目。FAST-LIO2 支持 raw LiDAR points 直接 scan-to-map 和 ikd-Tree 增量地图维护；**默认项目不以 GN10 那样的预建静态 ESDF 全局重定位为目标**。

本条目为 [导航·SLAM·自动驾驶栈 21 仓索引](navigation_slam_autonomy_stack_catalog.md) 组成部分。

---

## 官方入口与工程边界

- 源码（ROS 1）：`src/laserMapping.cpp` 主循环、`src/preprocess.cpp` 点云处理、`src/IMU_Processing.hpp` 状态传播及去畸变。
- MID-360 启动：`roslaunch fast_lio mapping_mid360.launch`；配置 `config/mid360.yaml` 中的 `/livox/lidar` 与 `/livox/imu`，launch 中 `feature_extract_enable=0`。
- README 要求同步 LiDAR / IMU，并保留逐点时间戳；已知外参时建议关闭在线外参估计。Livox driver 工作空间须在编译和启动前 source。
- PCD 保存需控制累积时长：`pcd_save.interval=-1` 可能持续增长内存。
- 复核仅覆盖官方文档与主要源码调用，不将 README 的部署表现写成独立复现实验结果。

## 对 wiki 的映射

- 实体：[FAST-LIO](../../wiki/entities/fast-lio.md)
- GN10 对照：[GN10 PointCloud Localization](../../wiki/entities/gn10-pointcloud-localization.md)
- 总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 选型：[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
