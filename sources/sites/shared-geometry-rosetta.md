# Shared Geometry as a Rosetta Stone — 作者项目页

- **类型：** project site
- **URL：** <https://dominik-schnaus.github.io/unpaired-rosetta/>
- **关联论文：** [arXiv:2610.09411](https://arxiv.org/abs/2610.09411)
- **代码：** [官方实现](https://github.com/dominik-schnaus/unpaired-rosetta)（已开源，MIT License）
- **数据：** 项目页展示 MS COCO 等既有数据集实验；未列出单独发布的新数据集。
- **机构：** TU Munich、MCML、Ulm University、ETH Zurich、MIT
- **入库日期：** 2026-10-22
- **一句话说明：** 论文作者的交互项目页，解释无配对嵌入对齐方法、模型组合、评估指标及跨模态演示。

## 页面核验摘录

- **设定：** 两组嵌入分别来自不同且不重叠的数据子集，编码器独立训练；示例使用 DINOv2 B/14 与 Qwen3-8B。方法对所有样本学习一个全局正交映射。
- **流程：** 分别聚类并匹配簇中心的相似结构，以 Gromov-Wasserstein 读出初始对应，之后交替更新 Hungarian 样本匹配与正交 Procrustes 映射。
- **指标：** 项目页将 FOSCTTM 定义为比真匹配更近的错误样本比例，越低越好；随机水平为 0.5。验证集为训练中未见的 COCO 验证配对。
- **关键分析：** 84 种视觉模型、语言模型和数据集组合中，CKA 解释了 73% 的对齐质量变化（R²=0.73，Spearman ρ=-0.90）。该预测分析使用配对样本，但方法本身不使用配对样本进行无监督训练。
- **少样本结果：** 已知配对数不超过 20 时，项目页报告该方法的 FOSCTTM 比七种配对方法中的最佳者低 14–28 倍；这些结果依赖论文所列数据与设定，不能外推为任意编码器组合都可对齐。

## 与仓库关系

- [统一实体页](../../wiki/entities/paper-shared-geometry-rosetta.md)
- [论文来源归档](../papers/shared_geometry_rosetta_arxiv_2610_09411.md)
- [代码来源归档](../repos/unpaired-rosetta.md)
