# CMU MSCV Semantic 3D Mapping（F23 Team 17）

> 来源归档

- **标题：** Semantic 3D Mapping — MSCV Student Project（CMU RI）
- **类型：** site（课程/项目页）
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）MSCV
- **链接：** https://mscvprojects.ri.cmu.edu/f23team17/sample-page/
- **项目索引：** https://mscvprojects.ri.cmu.edu/
- **入库日期：** 2026-07-26
- **一句话说明：** 用 DETR 做二维检测框、以框提示 SAM 得实例 mask，再经相机外参与位姿把 2D 标签投影到 3D 点云，作稀疏 LiDAR（如 VLP-16）室内伪标注与检测管线说明。
- **沉淀到 wiki：** [GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

---

## 项目页要点（步骤 2.5 核查）

### 标注流水线（与「SAM 2D→3D」问题直接对应）

1. 采集 RGB 序列 + 对应 360° LiDAR。
2. **DETR** 对单帧 RGB 做目标检测。
3. 以检测框为 query，用 **SAM** 得到实例级 mask。
4. 用相机外参与机器人位姿，把 2D 标签映射到对应 3D 点，得到逐帧标注点云。

### 其它技术尝试

- 曾尝试 **Range Image** 单阶段实例分割；发现标定误差在 2D→3D 放大严重，更适合作早期融合线索而非主路径。
- 3D 检测侧：pillar + attention（PIFENET 等）在 JRDB / CODa（通道降采样）上评测。

## 开源状态

- **项目页文档**：方法论与伪标注框架公开在 MSCV 项目站。
- **截至 2026-07-26：** 页面未列出独立完整训练/部署 GitHub 作为「一键复现生产栈」；定位为 **课程项目解决方案说明**，勿与 `autonomy_stack_go2` 混为一谈。
- 若后续补充官方仓，应另建 `sources/repos/` 并回链本页。

## 对 wiki 的映射

- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) — 「SAM 如何从 2D 到 3D」的直接对照案例
