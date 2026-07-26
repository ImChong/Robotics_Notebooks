# OVO（Open-Vocabulary Online Semantic Mapping）

> 来源归档

- **标题：** Open-Vocabulary Online Semantic Mapping for SLAM（OVO）
- **类型：** repo
- **论文：** https://arxiv.org/abs/2411.15043
- **项目页：** https://tberriel.github.io/ovo/
- **代码：** https://github.com/tberriel/OVO
- **许可：** MIT
- **入库日期：** 2026-07-26
- **一句话说明：** 输入有位姿的 RGB-D 关键帧，在线跟踪三维实例并融合 CLIP 特征；官方支持 SAM 2，并可对接 ORB-SLAM / Gaussian-SLAM 类后端做端到端开放词汇在线建图（含回环场景）。
- **沉淀到 wiki：** [ovo-semantic-mapping](../../wiki/entities/ovo-semantic-mapping.md)（实体）；[GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

---

## README 要点

- 2D mask 初始化常用 **SAM 2**（亦兼容 SAM 1）。
- 跨关键帧跟踪 3D segment，按可见性聚合 CLIP；含学习式 CLIP merge。
- 可与 **Gaussian-SLAM**、**ORB-SLAM2** 等骨干集成（项目页强调可不依赖真值位姿/几何的端到端演示路径）。

## 开源状态

- **已开源**：`tberriel/OVO`（MIT）。
- **边界：** 主线假设 **RGB-D + 位姿/SLAM**；接到 GO2 的 L1 点云需另做深度/投影桥接。

## 对 wiki 的映射

- 实体：[ovo-semantic-mapping](../../wiki/entities/ovo-semantic-mapping.md)
- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)
- 相关实体：[orb-slam3](../../wiki/entities/orb-slam3.md)（同族视觉 SLAM 对照）
