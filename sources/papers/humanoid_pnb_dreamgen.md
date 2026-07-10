# DreamGen: Unlocking Generalization in Robot Learning through Video World Models

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** DreamGen: Unlocking Generalization in Robot Learning through Video World Models
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2505.12705>
- **入库日期：** 2026-07-10
- **一句话说明：** DreamGen 是一个简单而高效的四阶段流水线，通过神经轨迹（neural trajectories）——由视频世界模型生成的合成机器人数据——训练能跨行为、跨环境泛化的机器人策略。流程：① 用图像到视频生成模型；② 把模型适配到目标机器人本体，生成逼真合成视频；③ 用潜动作模型（latent action model）或逆动力学模型（inverse-dynamics model）从视频中恢复伪动作序列；④ 用这些数据训练策略。还提出 DreamGen Bench 评测视频生成质量。实验中，仅用单一取放任务、单一环境的遥操作数据，DreamGen 就让人形在已见与未见环境完成 22 种新行为，展示强行为与环境泛化，为超越人工采集地扩展机器人学习开辟新路径。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](../../wiki/entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md).

## 对 wiki 的映射

- [paper-notebook-dreamgen-unlocking-generalization-in-robot-learn](../../wiki/entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models/DreamGen__Unlocking_Generalization_in_Robot_Learning_through_Video_World_Models.html>
- 论文：<https://arxiv.org/abs/2505.12705>
