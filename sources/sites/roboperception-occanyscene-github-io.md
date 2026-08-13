# OccAnyScene 项目页（roboperception.github.io/OccAnyScene）

> 来源归档（ingest 配套站点）

- **URL：** <https://roboperception.github.io/OccAnyScene/>
- **标题：** OccAnyScene — Towards Unified Indoor-Outdoor 3D Occupancy Prediction
- **机构：** 南方科技大学 / 上海人工智能实验室 / 哈尔滨工业大学深圳 / 鹏城实验室
- **论文：** <https://arxiv.org/abs/2608.08696> — 归档见 [`sources/papers/occanyscene_arxiv_2608_08696.md`](../papers/occanyscene_arxiv_2608_08696.md)
- **配套仓库（占位）：** <https://github.com/RoboPerception/OccAnyScene> — [`sources/repos/occanyscene.md`](../repos/occanyscene.md)
- **入库日期：** 2026-08-13
- **一句话说明：** 官方落地页：跨室内外 3D 语义占据；像素视锥高斯（PFFA + FPGC）；Occ-ScanNet / SurroundOcc-nuScenes 联合训练数字与遮挡可视化。

## 公开信息要点（截至入库日）

| 项 | 状态 |
|----|------|
| **Paper / BibTeX** | 已链 arXiv:2608.08696 |
| **Demo 视频** | 有（室内房间 ↔ 户外街道同一模型） |
| **方法叙事** | 像素视锥为场景自适应几何单元；PFFA 聚合几何+视觉上下文；FPGC 按视锥参数化高斯中心与尺度 |
| **代码按钮** | 指向 [`RoboPerception/OccAnyScene`](https://github.com/RoboPerception/OccAnyScene)；**仓内无可运行训练/推理**（见 repo 归档） |
| **结论** | 项目页可用于方法理解与定性结果；**复现代码待录用后发布** |

## 页面结构速记

1. **跨场景定位** — 一模型覆盖房间尺度细粒度与街道尺度大范围。
2. **结果摘要（页面对齐论文 DAv3）** — 室内 scene-specific 59.92% mIoU vs cross-scene 59.51%（-0.41）；户外 23.06% vs 22.87%（-0.19）。
3. **遮挡推理** — 表面相对深度增量把部分高斯推到前景遮挡物后方。
4. **效率** — 宣称 DAv2 在对比方法中参数最少、时延与显存较低。

## 关联资料

- 论文摘录：[`sources/papers/occanyscene_arxiv_2608_08696.md`](../papers/occanyscene_arxiv_2608_08696.md)
- 占位仓：[`sources/repos/occanyscene.md`](../repos/occanyscene.md)
- Wiki 实体：[`wiki/entities/paper-occanyscene.md`](../../wiki/entities/paper-occanyscene.md)
