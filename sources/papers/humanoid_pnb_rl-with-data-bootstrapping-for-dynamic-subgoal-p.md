# Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Reinforcement Learning with Data Bootstrapping for Dynamic Subgoal Pursuit in Humanoid Robot Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2506.02206>
- **入库日期：** 2026-07-10
- **一句话说明：** 安全、实时导航是人形应用的基础，但现有双足导航框架常难以平衡计算效率与稳定行走所需的精度。本文提出一个分层框架，持续生成动态子目标引导机器人穿越杂乱环境：高层 RL 规划器在机器人中心坐标系里选子目标，低层基于 MPC 的规划器产出鲁棒行走步态去到达这些子目标。为加速并稳定训练，引入一种数据自举（data bootstrapping）技术——借基于模型的导航方法生成多样、信息丰富的数据集。在 Agility Digit 人形上、多种随机障碍场景仿真验证：相比原模型法与其它学习法，成功率与适应性显著提升。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-rl-with-data-bootstrapping-for-dynamic-subgoal-p](../../wiki/entities/paper-notebook-rl-with-data-bootstrapping-for-dynamic-subgoal-p.md).

## 对 wiki 的映射

- [paper-notebook-rl-with-data-bootstrapping-for-dynamic-subgoal-p](../../wiki/entities/paper-notebook-rl-with-data-bootstrapping-for-dynamic-subgoal-p.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation/RL_with_Data_Bootstrapping_for_Dynamic_Subgoal_Pursuit_in_Humanoid_Navigation.html>
- 论文：<https://arxiv.org/abs/2506.02206>
