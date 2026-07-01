# Ultra-Fusion

> 来源归档

- **标题：** Ultra-Fusion
- **类型：** repo
- **链接：** https://github.com/sjtuyinjie/Ultra-Fusion
- **Stars：** ~469（2026-07）
- **入库日期：** 2026-07-01
- **一句话说明：** Ultra-Fusion 官方实现：统一滑窗多传感器紧耦合 SLAM（WIO/VIO/LIO/LVIO + 可选轮速/GNSS），含因子级可靠性调度与在线时空标定。
- **沉淀到 wiki：** [paper-ultra-fusion-multi-sensor-slam](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)、[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

---

## 核心定位

论文 *Ultra-Fusion: A Resilient Tightly-Coupled Multi-Sensor Fusion SLAM Framework under Sensor Degradation and Spatiotemporal Perturbation for Intelligent Transportation Systems*（arXiv:2606.21223）的官方代码仓库。面向 **智能交通系统（ITS）** 中轮式、腿式与低空 UAV 的 **韧性定位与建图**：在单一滑窗因子图内融合 LiDAR、相机、IMU、轮速与 GNSS，并支持 **可观测性感知初始化**、**因子级可靠性调度（FRS）** 与 **在线时空标定（OSC）**。

- **项目页：** <https://sjtuyinjie.github.io/ultrafusion-web/>
- **论文：** <https://arxiv.org/abs/2606.21223>
- **配套基准：** [M3DGR](m3dgr.md)

---

## 对 wiki 的映射

- 实体页：[paper-ultra-fusion-multi-sensor-slam](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)
- 选型对比：[lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- 概念：[sensor-fusion](../../wiki/concepts/sensor-fusion.md)
- 论文 source：[ultra_fusion_arxiv_2606_21223.md](../papers/ultra_fusion_arxiv_2606_21223.md)
