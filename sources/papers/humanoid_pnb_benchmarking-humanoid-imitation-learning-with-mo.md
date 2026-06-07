# Benchmarking Humanoid Imitation Learning with Motion Difficulty

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** Benchmarking Humanoid Imitation Learning with Motion Difficulty
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty.html>
- **分类：** 11_Simulation_Benchmark
- **arXiv：** <https://arxiv.org/abs/2512.07248>
- **入库日期：** 2026-06-07
- **一句话说明：** 现有人形模仿学习的指标（如关节位置误差 MPJPE）只衡量「策略学得多像」，却没法告诉你「这段动作本身有多难」——本文用刚体动力学给出一个与策略无关的 Motion Difficulty Score (MDS)：对参考姿态做小扰动后看产生的力矩变化空间，从体积 / 方差 / 时间变化率三个维度算难度；再用 MDS 把 AMASS 重新切成难度分层的 MD-AMASS，并配套两个新指标 MID（最大可模仿难度） 与 DSJE（按难度分层的关节误差）——首次把「比 SOTA」变成「在每个难度档分别比 SOTA」。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-benchmarking-humanoid-imitation-learning-with-mo](../../wiki/entities/paper-notebook-benchmarking-humanoid-imitation-learning-with-mo.md).

## 对 wiki 的映射

- [paper-notebook-benchmarking-humanoid-imitation-learning-with-mo](../../wiki/entities/paper-notebook-benchmarking-humanoid-imitation-learning-with-mo.md)
- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../../wiki/overview/paper-notebook-category-11-simulation-benchmark.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty/Benchmarking_Humanoid_Imitation_Learning_with_Motion_Difficulty.html>
- 论文：<https://arxiv.org/abs/2512.07248>
