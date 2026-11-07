# SplaTAM — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** SplaTAM
- **类型：** repo
- **链接：** https://github.com/spla-tam/SplaTAM
- **入库日期：** 2026-11-08
- **一句话说明：** 面向 RGB-D 稠密 SLAM 的三维高斯跟踪与建图。
- **核对范围：** 完整阅读作者官方 README 和论文 https://arxiv.org/html/2312.02126 的方法、实验及限制，并核对主程序/Mapper/Tracker/rendering 代码入口：README：python scripts/splatam.py configs/iphone/splatam.py；scripts/splatam.py::get_dataset、initialize_first_timestep、get_pointcloud、initialize_optimizer；diff_gaussian_rasterization 渲染，utils/slam_external.py::densify / prune_gaussians 管理显式高斯。；未独立运行实测。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [SplaTAM 源码](https://github.com/spla-tam/SplaTAM) **（本归档主类型）**

## 官方源码结构与阅读要点

README：python scripts/splatam.py configs/iphone/splatam.py；scripts/splatam.py::get_dataset、initialize_first_timestep、get_pointcloud、initialize_optimizer；diff_gaussian_rasterization 渲染，utils/slam_external.py::densify / prune_gaussians 管理显式高斯。

在首帧 RGB-D 反投影初始化球形/各向同性 Gaussians；后续每帧先使用 3D Gaussian splatting 可微渲染 RGB、深度和 silhouette，利用 silhouette>阈值的已有地图像素只优化相机位姿，避免未观测区域干扰；再据 silhouette 与深度差把新出现的 RGB-D 表面反投影为新 Gaussians，随后以当前帧和有重叠关键帧优化高斯位置、颜色、尺度、不透明度。每帧 Tracking→Gaussian Densification→Map Updating 是作者 §3 明确顺序。

需要标定过的 RGB-D 相机和相对可靠的深度；大运动、跟踪失败或位姿误差会污染高斯地图；渲染 FPS 不等于完整 SLAM 系统速度。官方代码中基于 torch 和 CUDA renderer 的设置、密度更新与 keyframe 选择必须匹配具体数据集。

## 方法与边界

面向 RGB-D 稠密 SLAM 的三维高斯跟踪与建图。本归档已核对原始 README 与源码主要入口，但没有运行 GPU 的 SLAM 性能复现。

**已核对 README：** 官方提供 `bash_scripts/online_demo.bash` 和 `configs/iphone/online_demo.py` 路径，涉及 iPhone 采集与离线 RGB-D 数据；不能将其训练或重建能力未经测试直接套用任意传感器。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已按原始论文逐节核对方法、实验与局限。
- [x] 已核对 README/源码运行入口和关键 Tracking/Mapping 关系。
- [x] [SplaTAM 深读实体](../../wiki/entities/paper-splatam.md) 包含两个 Mermaid 图，健康度待 CI 验证。
