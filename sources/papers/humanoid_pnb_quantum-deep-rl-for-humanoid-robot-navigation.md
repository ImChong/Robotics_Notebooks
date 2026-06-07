# Quantum deep reinforcement learning for humanoid robot navigation task

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Quantum deep reinforcement learning for humanoid robot navigation task
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2509.11388>
- **入库日期：** 2026-06-07
- **一句话说明：** 本文提出 变分量子 Soft Actor-Critic（QuantumSAC）：在经典 SAC 框架里，把 actor 的核心网络换成参数化量子电路（PQC，编码电路 + 变分电路），再用一层经典网络把量子测量结果映射成连续动作的均值与方差，从而不依赖传统建图 / 规划，直接在高维状态空间里学控制——并首次在 MuJoCo 的 Humanoid-v4 / Walker2d-v4 这类大观测、大动作的人形 / 双足任务上跑通量子深度 RL。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-quantum-deep-rl-for-humanoid-robot-navigation](../../wiki/entities/paper-notebook-quantum-deep-rl-for-humanoid-robot-navigation.md).

## 对 wiki 的映射

- [paper-notebook-quantum-deep-rl-for-humanoid-robot-navigation](../../wiki/entities/paper-notebook-quantum-deep-rl-for-humanoid-robot-navigation.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation/Quantum_Deep_RL_for_Humanoid_Robot_Navigation.html>
- 论文：<https://arxiv.org/abs/2509.11388>
