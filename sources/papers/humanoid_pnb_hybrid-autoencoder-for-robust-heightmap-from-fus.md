# A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Hybrid_Autoencoder_for_Robust_Heightmap_from_Fused_Lidar_and_Depth_Data/Hybrid_Autoencoder_for_Robust_Heightmap_from_Fused_Lidar_and_Depth_Data.html>
- **分类：** 05_Locomotion
- **arXiv：** <https://arxiv.org/abs/2602.05855>
- **入库日期：** 2026-06-07
- **一句话说明：** 人形 perceptive locomotion 的"上游 = 地形感知"长期被「单传感器手工管线（深度图 / 体素栅格 / 占据图）」卡住——深度相机视场窄、近距离精但远处糊，LiDAR 反过来；本文提出用一个 hybrid Encoder-Decoder（CNN 抽空间 + GRU 维持时序）把深度图 + LiDAR（球面投影） + IMU 三路数据直接学到一个以机器人为中心的统一 2.5D 高度图上，让下游 locomotion 策略只面对一种规范化的地形表征，融合相比单模态把重建误差降了 7.2–9.9 %，3.2 s 时序窗口把地图漂移也压了下来。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-hybrid-autoencoder-for-robust-heightmap-from-fus](../../wiki/entities/paper-notebook-hybrid-autoencoder-for-robust-heightmap-from-fus.md).

## 对 wiki 的映射

- [paper-notebook-hybrid-autoencoder-for-robust-heightmap-from-fus](../../wiki/entities/paper-notebook-hybrid-autoencoder-for-robust-heightmap-from-fus.md)
- 分类父节点：[paper-notebook-category-05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Hybrid_Autoencoder_for_Robust_Heightmap_from_Fused_Lidar_and_Depth_Data/Hybrid_Autoencoder_for_Robust_Heightmap_from_Fused_Lidar_and_Depth_Data.html>
- 论文：<https://arxiv.org/abs/2602.05855>
