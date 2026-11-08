# Gaussian-SLAM — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** Gaussian-SLAM
- **类型：** repo
- **链接：** https://github.com/VladimirYugay/Gaussian-SLAM
- **入库日期：** 2026-11-08
- **一句话说明：** 使用高斯表示的稠密建图与相机追踪。
- **核对范围：** 完整阅读作者官方 README 和论文 https://arxiv.org/html/2312.10070 的方法、实验及限制，并核对主程序/Mapper/Tracker/rendering 代码入口：README：python run_slam.py configs/Replica/room0.yaml --input_path ... --output_path ...；run_slam.py 装配 GaussianSLAM，src/entities/gaussian_slam.py::run 调用 src/entities/tracker.py 和 mapper.py；mapper.py::compute_seeding_mask/seed_new_gaussians/optimize_submap。；未独立运行实测。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [Gaussian-SLAM 源码](https://github.com/VladimirYugay/Gaussian-SLAM) **（本归档主类型）**

## 官方源码结构与阅读要点

README：python run_slam.py configs/Replica/room0.yaml --input_path ... --output_path ...；run_slam.py 装配 GaussianSLAM，src/entities/gaussian_slam.py::run 调用 src/entities/tracker.py 和 mapper.py；mapper.py::compute_seeding_mask/seed_new_gaussians/optimize_submap。

主循环对每帧 RGB-D 先用已有子地图可微渲染相机视角的 color/depth，再结合光度和几何损失直接优化相机位姿；Mapper 用透明度与深度误差构造新高斯 seeding mask，对已知区域做稀疏采样更新。地图分为独立 submaps，只优化活跃子图，其他子图可卸载；区分场景大小和地图局部开销。论文 §3.2/3.3/3.4 均有明确算法。

分块子地图存在跨边界地图一致性和全局回环的工程挑战；深度质量、初始位姿和 RGB-D 数据同步会改变优化；GPU 上 Gaussian rasterizer 非完全确定，必须报告重复 seed 的波动。未经测试不可把论文渲染质量解释为机器人在任意低纹理环境里成功定位。

## 方法与边界

使用高斯表示的稠密建图与相机追踪。本归档已核对原始 README 与源码主要入口，但没有运行 GPU 的 SLAM 性能复现。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已按原始论文逐节核对方法、实验与局限。
- [x] 已核对 README/源码运行入口和关键 Tracking/Mapping 关系。
- [x] [Gaussian-SLAM 深读实体](../../wiki/entities/paper-gaussian-slam.md) 包含两个 Mermaid 图，健康度待 CI 验证。
