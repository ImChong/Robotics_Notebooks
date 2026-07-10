# EgoDemoGen: Egocentric Demonstration Generation for Viewpoint Generalization in Robotic Manipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** EgoDemoGen: Egocentric Demonstration Generation for Viewpoint Generalization in Robotic Manipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2509.22578>
- **入库日期：** 2026-07-10
- **一句话说明：** 基于模仿学习的视觉运动策略表现强，但常对第一视角视角变化（egocentric viewpoint shifts）敏感。EgoDemoGen 是一个框架，在无需多视角数据的前提下，生成新第一视角下的「观测-动作」配对演示。它由两部分组成：① EgoTrajTransfer——用运动技能分割 + 几何感知变换 + 逆运动学滤波，把机器人轨迹迁移到新第一视角帧；② EgoViewTransfer——一个条件视频生成模型，把新视角重投影的场景与渲染的机器人运动融合，合成逼真观测。实验：仿真策略成功率绝对提升 +24.6% 与 +16.9%；真机在不同视角条件下提升 +16.0% 与 +23.0%。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-egodemogen-novel-egocentric-demonstration-genera](../../wiki/entities/paper-notebook-egodemogen-novel-egocentric-demonstration-genera.md).

## 对 wiki 的映射

- [paper-notebook-egodemogen-novel-egocentric-demonstration-genera](../../wiki/entities/paper-notebook-egodemogen-novel-egocentric-demonstration-genera.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization/EgoDemoGen__Egocentric_Demonstration_Generation_for_Viewpoint_Generalization.html>
- 论文：<https://arxiv.org/abs/2509.22578>
