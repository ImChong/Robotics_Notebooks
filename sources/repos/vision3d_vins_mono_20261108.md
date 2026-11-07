# VINS-Mono — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** VINS-Mono
- **类型：** repo
- **链接：** https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- **入库日期：** 2026-11-08
- **一句话说明：** 单目视觉与 IMU 的优化式状态估计。
- **核对范围：** 已完整阅读官方 README 的版本/依赖、EuRoC/AR、地图保存/重用、在线相机-IMU 外参/时间标定、rolling-shutter 与源码入口；核对 launch/euroc.launch、feature_tracker_node.cpp、estimator_node.cpp/estimator.cpp、pose_graph_node.cpp 的 ROS 回调连接。未在本机运行。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) **（本归档主类型）**

## 官方系统模块和运行事实

- `roslaunch vins_estimator euroc.launch` 从 launch 加载 `feature_tracker`、`vins_estimator`、`pose_graph` 三个节点；RViz 独立启动，EuRoC bag 单独播放。
- FeatureTracker 使用 KLT 光流特征/角点补充，以 `feature` 点云话题发布；`vins_estimator/src/estimator_node.cpp` 接受 IMU+视觉特征，`Estimator::processIMU/processImage` 做 IMU 预积分、滑窗非线性优化和失败重启。
- `pose_graph/src/pose_graph_node.cpp` 管理 keyframes/BRIEF/DBoW2 候选和优化；图优化是在全局 x,y,z,yaw 4 DoF，不重做完全 6DoF 重力方向。
- README 要求 ROS Kinetic/Ubuntu16.04 和 Ceres **1.14**，不保证 Ceres2.0+/ROS2 Humble 兼容。需图像 >20Hz、IMU >100Hz、稳定 timestamp；可 `estimate_extrinsic=2` 在线粗外参并 `estimate_td=1` 估计时差。

## 方法与边界

单目视觉与 IMU 的优化式状态估计。已阅读作者 README 和关键 ROS 入口，论文实验与完整原理单列来源，未在新环境实机复现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 参见配套论文通读记录。
- [x] 核对 launch、feature tracker、estimator、pose graph 主节点与 ROS callback。
- [x] 已新增独立 VINS-Mono 节点，论文流程与代码时序图完整。
