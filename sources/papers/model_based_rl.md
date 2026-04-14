# model_based_rl

> 来源归档（ingest）

- **标题：** Model-Based Reinforcement Learning — 世界模型与规划
- **类型：** paper
- **来源：** arXiv / NeurIPS / ICLR
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 MBRL 代表算法（Dreamer / MBPO / PETS / TD-MPC），支撑 model-based-rl.md wiki 页面。

## 核心论文摘录（MVP）

### 1) Mastering Diverse Domains through World Models (DreamerV3, Hafner et al., 2023)
- **链接：** <https://arxiv.org/abs/2301.04104>
- **核心贡献：** 单一算法在 Atari / DMControl / Minecraft / 足式机器人等 150+ 任务上 SOTA，证明世界模型的通用性。RSSM（循环状态空间模型）在潜空间中编码世界动态。
- **关键架构：**
  ```
  RSSM: z_t = f(z_{t-1}, a_{t-1}, o_t)
  Actor/Critic: 在潜空间想象轨迹，无需真实交互
  ```
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 2) When to Trust Your Model: Model-Based Policy Optimization (MBPO, Janner et al., 2019)
- **链接：** <https://arxiv.org/abs/1906.08253>
- **核心贡献：** 短 rollout（1~5 步）+ 集成神经网络模型 + SAC，用约 5% 的 SAC 样本量达到相同性能，给出何时使用模型 rollout 的理论分析。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 3) Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS, Chua et al., 2018)
- **链接：** <https://arxiv.org/abs/1805.12114>
- **核心贡献：** 概率集成模型 + CEM 规划，无需策略网络，在机器人操作真实场景下仅需少量交互即可有效学习。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)

### 4) TD-MPC2: Scalable, Robust World Models for Continuous Control (Hansen et al., 2023)
- **链接：** <https://arxiv.org/abs/2310.16828>
- **核心贡献：** 潜空间动力学模型 + MPPI 规划 + TD 价值函数，单一模型在 80+ 连续控制任务（含足式）上超越 SAC/TD3，训练简单可扩展。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 5) Dyna, an Integrated Architecture for Learning, Planning, and Reacting (Sutton, 1991)
- **链接：** <https://dl.acm.org/doi/10.1145/122344.122377>
- **核心贡献：** 最早将模型生成的"虚拟经验"和真实经验结合训练价值函数，奠定了现代 MBRL 的 Dyna 架构范式。
- **对 wiki 的映射：**
  - [Model-Based RL](../../wiki/methods/model-based-rl.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] DreamerV3 / MBPO / PETS / TD-MPC2 / Dyna 五条核心摘要
- [ ] 后续补：MBRL 在机器人上的落地挑战（高频控制 vs 规划开销）
- [ ] 后续补：与 Model-Free RL 样本效率数值对比表
