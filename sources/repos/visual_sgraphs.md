# visual_sgraphs（vS-Graphs）

> 来源归档

- **标题：** visual_sgraphs / vS-Graphs
- **类型：** repo
- **链接：** <https://github.com/snt-arg/visual_sgraphs>
- **论文：** arXiv:2503.01783（RA-L 2026）
- **Stars：** ~50+（2026-07）
- **入库日期：** 2026-07-02
- **最近核对：** 2026-11-21
- **一句话说明：** 基于 **ORB-SLAM 3.0** 的 ROS2 Jazzy RGB-D / RGB-D-Inertial VSLAM 仓库，提供在线分层 3D 场景图（墙/地面 → 房间/楼层）、可选 Voxblox 与 ArUco 扩展、Docker 环境与评测脚本。
- **沉淀到 wiki：** [paper-vs-graphs-visual-slam-scene-graph](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

---

## 核心定位与仓库结构（2026-11-21 核查）

- **运行入口：** `launch/rgbd.launch.py`、`launch/rgbd-imu.launch.py`（对应 ROS2 Jazzy + Ubuntu 24.04.2 LTS，`src/ros_rgbd.cc` / `src/ros_rgbd_inertial.cc` 与 `core/` 核心库）
- **配置与环境：** `config/system_params.yaml`、`Dependencies.md`、`docker/` 可复现镜像、`evaluation/` 评测脚本
- **新增线程：** Building Component Recognition（`pFCN/YOSO` 全景分割 + RANSAC 平面拟合）、Structural Element Recognition（房间/楼层推断，可选 Voxblox 自由空间聚类与 `aruco_ros` 语义命名）
- **输入：** RGB-D / RGB-D-IMU
- **输出：** 位姿轨迹 + 几何地图 + 分层 3D 场景图
- **相关项目：** [LiDAR S-Graphs](https://github.com/snt-arg/lidar_situational_graphs)（同团队激光版）

---

## 对 wiki 的映射

- 论文实体：[paper-vs-graphs-visual-slam-scene-graph](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)
- 基线对照：[orb-slam3](../../wiki/entities/orb-slam3.md)
- 导航栈总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
