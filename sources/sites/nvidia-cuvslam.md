# NVIDIA cuVSLAM 项目文档站

> 来源归档

- **标题：** cuVSLAM: CUDA-Accelerated Visual Odometry and Mapping
- **类型：** site（NVIDIA Isaac 文档站）
- **URL：** <https://nvidia-isaac.github.io/cuVSLAM/>
- **论文：** <https://arxiv.org/abs/2506.04359>
- **入库日期：** 2026-09-24
- **一句话说明：** cuVSLAM 官方文档：PyCuVSLAM / C++ API、跟踪模式、性能与 ROS 2 接入指针。

## 开源核查（步骤 2.5，2026-09-24）

| 链接 | 状态 |
|------|------|
| [GitHub — nvidia-isaac/cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) | **已开源**（NVIDIA Community License；PyCuVSLAM 预编译 wheel + C++ 库） |
| [Isaac ROS Visual SLAM](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) | **已开源**（ROS 2 封装，依赖 cuVSLAM） |
| [arXiv:2506.04359](https://arxiv.org/abs/2506.04359) | 技术报告 PDF / HTML |

项目页与 README 均链到上述仓库；无「待发布」占位叙述。

## 页面要点（策展）

- **定位：** CUDA 加速视觉里程计 / VSLAM 库；1–32 路相机、可选 IMU / RGB-D；面向 Jetson 与 x86+GPU 实时栈。
- **API：** [Python](https://nvidia-isaac.github.io/cuVSLAM/python/) / [C++](https://nvidia-isaac.github.io/cuVSLAM/cpp/) 文档；预编译 **PyCuVSLAM** wheel（Ubuntu 22.04/24.04、Jetson Orin/Thor）。
- **模式：** Mono / RGBD / Multicamera / Inertial（立体 VIO）/ Multisensor（实验性，需 cuNLS 构建）。
- **ROS 2：** 经 Isaac ROS Visual SLAM 节点对接 Nav2 / nvblox 等感知链。

## 对 wiki 的映射

- [paper-cuvslam](../../wiki/entities/paper-cuvslam.md)
- [cuvslam 论文摘录](../papers/cuvslam_arxiv_2506_04359.md)
- [cuvslam 代码归档](../repos/cuvslam.md)
- [isaac-ros-visual-slam](../../wiki/entities/isaac-ros-visual-slam.md)
