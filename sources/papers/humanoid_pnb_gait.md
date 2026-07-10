# GAIT: Legged Robot Proprioceptive State Estimation with Attention over Inertial-Leg Tokens

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** GAIT: Legged Robot Proprioceptive State Estimation with Attention over Inertial-Leg Tokens
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia.html>
- **分类：** 09_State_Estimation
- **arXiv：** <https://arxiv.org/abs/2606.14160>
- **入库日期：** 2026-07-10
- **一句话说明：** GAIT 把足式机器人的本体感知测量做成「惯性-腿部（Inertial-Leg, IL）分词」——IMU 是一路 token、每条腿各是一路 token——再用轻量的 Perceiver IO 交叉注意力 学习「哪路测量此刻更可信」。因为一条腿只有触地时的前向运动学速度才可靠，注意力天然学到了「按接触状态重新赋权」这件事，不需要显式接触估计器。网络只用 trot（对角小跑） 数据训练，却能泛化到 bound / pace / pronk 等未见步态，并把估出的机身速度喂进 IEKF 得到完整位姿——同时推理只需 0.12 MFLOPs、0.7 ms。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-gait](../../wiki/entities/paper-notebook-gait.md).

## 对 wiki 的映射

- [paper-notebook-gait](../../wiki/entities/paper-notebook-gait.md)
- 分类父节点：[paper-notebook-category-09-state-estimation](../../wiki/overview/paper-notebook-category-09-state-estimation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/09_State_Estimation/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia/GAIT__Legged_Robot_Proprioceptive_State_Estimation_with_Attention_over_Inertia.html>
- 论文：<https://arxiv.org/abs/2606.14160>
