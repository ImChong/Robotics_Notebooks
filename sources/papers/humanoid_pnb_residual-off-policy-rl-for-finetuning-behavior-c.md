# Residual Off-Policy RL for Finetuning Behavior Cloning Policies

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Residual Off-Policy RL for Finetuning Behavior Cloning Policies
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2509.19301>
- **入库日期：** 2026-07-10
- **一句话说明：** 行为克隆（BC）能学到不错的视觉运动策略，但受限于人类演示质量、采集人力、离线数据的边际收益递减。强化学习（RL）靠自主交互、潜力大，但直接在真机训 RL 难——样本低效、安全、稀疏奖励长时程，对高自由度（DoF）系统尤甚。本文给出一个把 BC 与 RL 优点结合的残差学习配方：把 BC 策略当黑盒基座，用样本高效的离策略 RL 学每步的轻量残差修正。方法只需稀疏二值奖励，即可在高自由度系统（仿真与真机）上改进操作策略。尤其，作者据其所知首次在带灵巧手的人形真机上成功进行 RL 训练，在多项视觉任务上取得 SOTA，指向把 RL 真正部署到现实的可行路径。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-residual-off-policy-rl-for-finetuning-behavior-c](../../wiki/entities/paper-notebook-residual-off-policy-rl-for-finetuning-behavior-c.md).

## 对 wiki 的映射

- [paper-notebook-residual-off-policy-rl-for-finetuning-behavior-c](../../wiki/entities/paper-notebook-residual-off-policy-rl-for-finetuning-behavior-c.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies/Residual_Off-Policy_RL_for_Finetuning_Behavior_Cloning_Policies.html>
- 论文：<https://arxiv.org/abs/2509.19301>
