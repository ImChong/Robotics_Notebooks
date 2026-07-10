# Learning to Look: Seeking Information for Decision Making via Policy Factorization

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Learning to Look: Seeking Information for Decision Making via Policy Factorization
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2410.18964>
- **入库日期：** 2026-07-10
- **一句话说明：** 许多操作任务需要主动或交互式探索才能成功——智能体要主动寻找每一阶段所需的信息（如移动机器人的头去找操作相关信息；或多机器人里一个侦察机器人为另一个找信息）。本文把这类任务刻画为一种新问题：因子化上下文马尔可夫决策过程（factorized Contextual MDP），并提出 DISaM ——一个双策略解法：① 信息寻求策略（information-seeking）探索环境找到相关上下文信息；② 信息接收策略（information-receiving）利用上下文达成操作目标。这种因子化让两策略可分开训练（用接收策略给寻求策略提供奖励）。测试时，双智能体按操作策略对"下一步最佳动作"的不确定性来平衡探索与利用。在五个需信息寻求的操作任务（仿真 + 真机）上，DISaM 大幅优于已有方法。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-learning-to-look-seeking-information-for-decisio](../../wiki/entities/paper-notebook-learning-to-look-seeking-information-for-decisio.md).

## 对 wiki 的映射

- [paper-notebook-learning-to-look-seeking-information-for-decisio](../../wiki/entities/paper-notebook-learning-to-look-seeking-information-for-decisio.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization.html>
- 论文：<https://arxiv.org/abs/2410.18964>
