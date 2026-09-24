# cuVSLAM：CUDA 加速视觉里程计与建图（arXiv:2506.04359）

> 论文来源归档（ingest）

- **标题：** cuVSLAM: CUDA accelerated visual odometry and mapping
- **类型：** paper / visual-slam / vio / cuda / edge-computing
- **arXiv：** <https://arxiv.org/abs/2506.04359> · PDF：<https://arxiv.org/pdf/2506.04359>
- **作者：** Alexander Korovko, Dmitry Slepichev, Alexander Efitorov, Aigul Dzhumamuratova, Viktor Kuznetsov, Hesam Rabeti, Joydeep Biswas, Soha Pouya
- **机构：** NVIDIA（英伟达）
- **项目页：** <https://nvidia-isaac.github.io/cuVSLAM/>
- **代码：** <https://github.com/nvidia-isaac/cuVSLAM>
- **ROS 2：** <https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam>
- **入库日期：** 2026-09-24
- **一句话说明：** CUDA 全栈加速的模块化 VSLAM：前端低延迟里程计 + 异步后端 PGO/回环；1–32 相机与 VI/RGB-D 模式；KITTI / EuRoC 等基准上报告 SOTA 级精度与 Jetson 实时性。

## 核心摘录（面向 wiki 编译）

### 1) 前后端分离与「平滑里程计」前端

- **要点：** **Frontend** 专注在线位姿、高吞吐；维护 **局部里程计地图**（最近 N 关键帧），优先轨迹平滑、避免回环/PGO 引入的位姿跳变。**Backend** 异步处理全局一致地图、回环与位姿图优化。
- **对 wiki 的映射：** [`wiki/entities/paper-cuvslam.md`](../../wiki/entities/paper-cuvslam.md)

### 2) 2D 特征管线与 GPU 加速

- **要点：** 分块均匀 **关键点选择**、改进 **Lucas–Kanade** 跟踪、立体/多相机 **跨相机跟踪** 与三角化；关键帧触发 **局部稀疏 BA（SBA）**；CUDA 贯穿特征与 BA。
- **对 wiki 的映射：** 同上实体页

### 3) Multicamera 与 Frustum Intersection Graph（FIG）

- **要点：** 多相机模式下根据外参自动构建 **FIG**（视锥重叠有向图），在走廊等单方向特征贫乏场景用多视角约束位姿；多相机需 **硬件级同步**（如 RealSense 多机同步指南）。
- **对 wiki 的映射：** 同上；[`wiki/comparisons/lidar-slam-lio-vio-selection.md`](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

### 4) 传感器模式谱系

- **要点：** **Mono**（尺度模糊）、**RGBD** 稠密帧间、**Multicamera**（≥2 相机含一对立体）、**Inertial**（立体+VIO）、**Multisensor**（RGB/RGB-D 任意组合 + 可选 IMU，实验性、需 cuNLS 构建）。库默认 **免调参**，依赖准确 **内外参** 与 rig 配置。
- **对 wiki 的映射：** 同上；[`wiki/entities/isaac-ros-visual-slam.md`](../../wiki/entities/isaac-ros-visual-slam.md)

### 5) 基准与边缘部署

- **要点：** 技术报告称 KITTI 里程计 **平均轨迹误差 <1%**、EuRoC **位置误差 <5 cm**，并在 **Jetson** 上实时；**Multi-Stereo** 相对单立体在难序列上更鲁棒。工程侧强调标定、同步、帧率（~30 FPS 人速运动）、分辨率（VGA+）与运动模糊控制。
- **对 wiki 的映射：** 同上；[`wiki/overview/navigation-slam-autonomy-stack.md`](../../wiki/overview/navigation-slam-autonomy-stack.md)

## 对 wiki 的映射（仓库层）

- 代码归档：[sources/repos/cuvslam.md](../repos/cuvslam.md)
- 项目页：[sources/sites/nvidia-cuvslam.md](../sites/nvidia-cuvslam.md)
- ROS 封装：[sources/repos/isaac_ros_visual_slam.md](../repos/isaac_ros_visual_slam.md)

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 开源状态（项目页 + GitHub 已核）
- [x] 实体页与全站索引（ingest 任务内同步）
