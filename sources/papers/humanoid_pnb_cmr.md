# CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion.html>
- **分类：** 05_Locomotion
- **arXiv：** <https://arxiv.org/abs/2602.03511>
- **入库日期：** 2026-07-10
- **一句话说明：** 非结构地形上，人形机器人的传感器会出错、模型也不准，观测噪声容易在闭环里被放大成失稳。CMR 的思路是：与其在原始观测空间里硬抗噪声，不如把观测映射到一个「收缩」潜空间——在那里相邻状态的扰动会随时间逐步收缩衰减。它用对比学习（InfoNCE）保住任务相关的语义结构（防止收缩过度把有用信息也压没了），同时用 Lipschitz 正则显式逼策略满足收缩条件，二者写成一个辅助损失直接加进 PPO；理论上还证明了：当潜动态满足收缩性（κ<1），策略性能因噪声的退化被一个与时间步长无关的上界 O(η/(1−κ)) 框住，而非标准分析里随时域指数爆炸。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-cmr-contractive-mapping-embeddings-for-robust-hu](../../wiki/entities/paper-notebook-cmr-contractive-mapping-embeddings-for-robust-hu.md).

## 对 wiki 的映射

- [paper-notebook-cmr-contractive-mapping-embeddings-for-robust-hu](../../wiki/entities/paper-notebook-cmr-contractive-mapping-embeddings-for-robust-hu.md)
- 分类父节点：[paper-notebook-category-05-locomotion](../../wiki/overview/paper-notebook-category-05-locomotion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion/CMR__Contractive_Mapping_Embeddings_for_Robust_Humanoid_Locomotion.html>
- 论文：<https://arxiv.org/abs/2602.03511>
