# cuVSLAM

> 来源归档

- **标题：** cuVSLAM
- **类型：** repo
- **链接：** https://github.com/nvidia-isaac/cuVSLAM
- **Stars：** ~200+（2026-09，随发布增长）
- **入库日期：** 2026-09-24
- **一句话说明：** NVIDIA CUDA 加速视觉里程计 / SLAM 库：PyCuVSLAM wheel、C++ API、多相机 FIG 与前后端分离架构。
- **沉淀到 wiki：** [paper-cuvslam](../../wiki/entities/paper-cuvslam.md)、[isaac-ros-visual-slam](../../wiki/entities/isaac-ros-visual-slam.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

---

## 核心定位

**cuVSLAM** 是 NVIDIA 发布的 **视觉跟踪与 SLAM** 库（[arXiv:2506.04359](https://arxiv.org/abs/2506.04359)），通过 **CUDA** 在全管线（特征、跟踪、局部 BA）上实现 **边缘实时** 性能。支持 **单目 → 32 相机** 任意几何布局、立体 / 多立体 **Multicamera**、**RGB-D**、**立体 VIO** 与实验性 **Multisensor** 融合。

- **项目页 / 文档：** <https://nvidia-isaac.github.io/cuVSLAM/>
- **论文：** <https://arxiv.org/abs/2506.04359>
- **ROS 2 封装：** [isaac_ros_visual_slam](isaac_ros_visual_slam.md)
- **许可：** NVIDIA Community License（见仓库 `LICENSE`）

**工程入口（README）：**

- 快速开始：`pip` 安装 [Release wheel](https://github.com/nvidia-isaac/cuVSLAM/releases) → `examples/`
- 源码构建：仓库 `BUILD.md` / CMake
- ROS 2：NVIDIA-ISAAC-ROS/isaac_ros_visual_slam

---

## 对 wiki 的映射

- 实体页：[paper-cuvslam](../../wiki/entities/paper-cuvslam.md)
- ROS 封装：[isaac-ros-visual-slam](../../wiki/entities/isaac-ros-visual-slam.md)
- 总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 选型：[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- 论文 source：[cuvslam_arxiv_2506_04359.md](../papers/cuvslam_arxiv_2506_04359.md)
- 项目页：[nvidia-cuvslam.md](../sites/nvidia-cuvslam.md)
