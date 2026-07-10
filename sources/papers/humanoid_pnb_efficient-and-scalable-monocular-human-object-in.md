# Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2512.00960>
- **入库日期：** 2026-07-10
- **一句话说明：** 机器人要学操作离不开大规模、多样化的「人-物交互（HOI）」数据，但高精度动捕系统又贵又受限、采不到户外运动 / 工业作业这类真实场景。本文主张直接从普通单目互联网视频里抠出 4D HOI 数据：用稀疏接触标注把昂贵的逐帧密集标注降到「平均 6.7 个点 / 约 10 分钟一条」，再用 InterPoint 多模态预测器 + 4DHOISolver 两阶段优化把人、物、接触对齐成时空连贯且物理合理的轨迹，产出 Open4DHOI 数据集，并用 RL 动作模仿证明重建质量足以驱动仿真智能体复现交互动作。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-efficient-and-scalable-monocular-human-object-in](../../wiki/entities/paper-notebook-efficient-and-scalable-monocular-human-object-in.md).

## 对 wiki 的映射

- [paper-notebook-efficient-and-scalable-monocular-human-object-in](../../wiki/entities/paper-notebook-efficient-and-scalable-monocular-human-object-in.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html>
- 论文：<https://arxiv.org/abs/2512.00960>
