# imitation_learning

> 来源归档（ingest）

- **标题：** Imitation Learning（BC / DAgger / Diffusion）
- **类型：** paper
- **来源：** arXiv / conference
- **入库日期：** 2026-04-08
- **最后更新：** 2026-04-14
- **一句话说明：** 聚焦模仿学习核心算法与机器人动作生成路线，为 IL 页面持续提供可溯源输入。

## 核心论文摘录（MVP）

### 1) A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (Ross et al., 2011)
- **链接：** <https://proceedings.mlr.press/v15/ross11a.html>
- **核心贡献：** 提出 DAgger，系统解决 BC 的 covariate shift 问题。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 2) Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)
- **链接：** <https://arxiv.org/abs/2303.04137>
- **核心贡献：** 将 diffusion model 引入策略建模，在高维连续动作控制中提升行为质量与泛化性。
- **对 wiki 的映射：**
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 3) ACT: Action Chunking with Transformers (Zhao et al., 2023)
- **链接：** <https://arxiv.org/abs/2304.13705>
- **核心贡献：** 通过 action chunking 改善长时序机器人操作稳定性，降低闭环误差累积。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 4) ASE: Adversarial Skill Embeddings for Hierarchical RL (Peng et al., 2022)
- **链接：** <https://arxiv.org/abs/2205.01906>
- **核心贡献：** 把技能表征为可插值 latent，连接 IL 与 RL 的技能复用路径。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] 已补 DAgger / ACT / Diffusion Policy / ASE 的基础摘要
- [~] 后续补：增加“单步 BC vs 序列建模”对比表
