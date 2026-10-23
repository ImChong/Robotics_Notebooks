# Zero-to-CAD 100K 精选数据集

> 来源归档（dataset / Hugging Face；核查日期：2026-10-23）

- **数据集：** Zero-To-CAD-100k
- **发布方：** Autodesk AI Lab（Hugging Face 组织 `ADSKAILab`）
- **Hugging Face：** <https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k>
- **许可：** Apache-2.0（以数据集卡标注为准）
- **论文：** <https://arxiv.org/abs/2604.24479>（[论文来源](../papers/zero_to_cad_arxiv_2604_24479.md)）
- **配套模型：** <https://huggingface.co/ADSKAILab/Zero-To-CAD-Qwen3-VL-2B>（[模型归档](../models/zero-to-cad-qwen3-vl-2b.md)）
- **项目实体：** [Zero-to-CAD](../../wiki/entities/zero-to-cad.md)

## 数据卡核查

- 这是从完整约 **1M** 合成样本中按几何多样性选出的约 **100K** 子集：将 8 视角 DINOv3 特征聚类为 100K 组，再选取最接近各簇中心的代表样本。
- 卡片列出的划分为 **81,015 train、9,734 validation、9,767 test**，合计 **100,516** 条。
- 每条样本包括可执行 CadQuery 源码、8 张 256×256 渲染图、操作序列元数据，以及 STL / STEP 导出文件。
- 数据卡建议将 100K 用于原型、基准评测和算力受限的微调；大规模训练使用完整 1M 数据集。

## 对 Robotics_Notebooks 的映射

- [Zero-to-CAD 实体页](../../wiki/entities/zero-to-cad.md)
- 相邻概念：[文字生成 CAD](../../wiki/concepts/text-to-cad.md)
