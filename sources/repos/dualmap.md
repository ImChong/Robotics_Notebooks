# DualMap

> 来源归档

- **标题：** DualMap — Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **类型：** repo
- **论文：** https://arxiv.org/abs/2506.01950（RAL 2025）
- **项目页：** https://eku127.github.io/DualMap/
- **代码：** https://github.com/Eku127/DualMap
- **入库日期：** 2026-07-26
- **一句话说明：** 在线开放词汇语义建图；MobileCLIP + YOLO-World / MobileSAM / FastSAM 等混合前端；双地图（全局抽象 + 局部具体）；支持 ROS1/ROS2、rosbag 与动态场景自然语言导航。
- **沉淀到 wiki：** [dualmap](../../wiki/entities/dualmap.md)（实体）；[GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

---

## README 要点

- **输入模式：** Dataset / ROS（含 rosbag）/ Record3d（iPhone）。
- **前端：** 混合分割（YOLO-World 类检测 + FastSAM/MobileSAM 等开放分割）+ MobileCLIP 特征。
- **双地图：** 全局 abstract map 候选选择 + 局部 concrete map 精达目标；支持动态更新。
- **导航：** 自然语言查询驱动；可与 Habitat Data Collector 联调（ROS2）。

## 开源状态

- **已开源**：`Eku127/DualMap`（含文档与子模块说明；需 `--recurse-submodules` 拉 MobileCLIP）。
- **工程提示：** 适合作为 GO2 语义层候选；需自配相机/深度或 LiDAR 投影与时间同步，非官方 GO2 一体栈。

## 对 wiki 的映射

- 实体：[dualmap](../../wiki/entities/dualmap.md)
- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)
