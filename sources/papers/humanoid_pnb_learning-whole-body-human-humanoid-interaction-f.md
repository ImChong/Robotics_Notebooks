# Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Whole-Body_Human-Humanoid_Interaction_from_Human-Human_Demonstrations/Learning_Whole-Body_Human-Humanoid_Interaction_from_Human-Human_Demonstrations.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2601.09518>
- **入库日期：** 2026-07-10
- **一句话说明：** 让人形机器人与人发生物理交互是关键前沿，但人-人形交互（HHoI）数据极度稀缺。借用海量人-人交互（HHI）数据是可扩展替代，但作者发现：标准重定向会破坏交互中最关键的「接触」。为此提出 PAIR（Physics-Aware Interaction Retargeting）——一个以接触为中心的两阶段管线，跨形态差异保住接触语义，生成物理一致的 HHoI 数据。但高质量数据又暴露第二个失败：常规模仿学习只照搬轨迹、缺乏交互理解。于是再提出 D-STAR（Decoupled Spatio-Temporal Action Reasoner）——一个分层策略，把「何时动」与「何处动」解耦：相位注意力（Phase Attention）管时间、多尺度空间模块管空间，二者由扩散头融合，产出同步的全身行为而非简单模仿。解耦让模型学到鲁棒的时间相位而不被空间噪声干扰，带来响应式、同步的协作。仿真中显著优于基线，构成「从 HHI 数据学复杂全身交互」的完整有效流水线。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-learning-whole-body-human-humanoid-interaction-f](../../wiki/entities/paper-notebook-learning-whole-body-human-humanoid-interaction-f.md).

## 对 wiki 的映射

- [paper-notebook-learning-whole-body-human-humanoid-interaction-f](../../wiki/entities/paper-notebook-learning-whole-body-human-humanoid-interaction-f.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Learning_Whole-Body_Human-Humanoid_Interaction_from_Human-Human_Demonstrations/Learning_Whole-Body_Human-Humanoid_Interaction_from_Human-Human_Demonstrations.html>
- 论文：<https://arxiv.org/abs/2601.09518>
