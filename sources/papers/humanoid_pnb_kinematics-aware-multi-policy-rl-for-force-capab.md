# Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation.html>
- **分类：** 04_Loco-Manipulation_and_WBC
- **arXiv：** <https://arxiv.org/abs/2511.21169>
- **入库日期：** 2026-07-10
- **一句话说明：** 人形机器人有类人形态，在工业里潜力大。但现有 loco-manip 多聚焦灵巧操作，难满足高负载工业对「灵巧 + 主动力交互」的双重要求。本文提出一个基于 RL 的解耦三阶段训练流水线：上身策略、下身策略、delta 指令策略。为加速上身训练，设计一个启发式奖励——通过隐式嵌入前向运动学（FK）先验，让策略更快收敛且性能更优；为下身，开发一个基于力的课程学习策略，使机器人能主动施加并调节与环境的交互力。这样把「灵巧」与「主动发力」统一进同一框架，面向高负载工业搬运/推压等任务。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-kinematics-aware-multi-policy-reinforcement-lear](../../wiki/entities/paper-notebook-kinematics-aware-multi-policy-reinforcement-lear.md).

## 对 wiki 的映射

- [paper-notebook-kinematics-aware-multi-policy-reinforcement-lear](../../wiki/entities/paper-notebook-kinematics-aware-multi-policy-reinforcement-lear.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation.html>
- 论文：<https://arxiv.org/abs/2511.21169>
