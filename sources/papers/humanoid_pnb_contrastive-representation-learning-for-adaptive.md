# Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2509.12858>
- **入库日期：** 2026-06-07
- **一句话说明：** 主流人形 RL 行走面临两难选择——纯本体感知策略反应快但被动（只能"踩到了再调"），而依赖深度图/高程图的感知驱动策略主动但脆弱（深度噪声、外参漂移、视角遮挡都会让 sim-to-real 崩掉）。本文用对比学习把仿真侧的特权环境信息（地形高度、摩擦、质量、外力等）"蒸馏"到 actor 的隐状态里，同时引入一个自适应步态时钟让策略根据"已感知到但实际看不见"的地形主动调整步频，从而在不接任何外部感知模块的前提下兼具反应与主动性，全尺寸人形零样本通过 30 cm 台阶和 26.5° 斜坡。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-contrastive-representation-learning-for-adaptive](../../wiki/entities/paper-notebook-contrastive-representation-learning-for-adaptive.md).

## 对 wiki 的映射

- [paper-notebook-contrastive-representation-learning-for-adaptive](../../wiki/entities/paper-notebook-contrastive-representation-learning-for-adaptive.md)
- 分类父节点：[paper-notebook-category-10-sim-to-real](../../wiki/overview/paper-notebook-category-10-sim-to-real.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html>
- 论文：<https://arxiv.org/abs/2509.12858>
