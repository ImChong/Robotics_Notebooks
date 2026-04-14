# locomotion_rl

> 来源归档（ingest）

- **标题：** Locomotion RL（Humanoid / Legged）
- **类型：** paper
- **来源：** arXiv / conference / 开源仓库
- **入库日期：** 2026-04-08
- **最后更新：** 2026-04-14
- **一句话说明：** 聚焦人形/腿足机器人 locomotion 的强化学习代表工作，用于支撑 RL、locomotion、sim2real 页面的事实来源。

## 核心论文摘录（MVP）

### 1) AMP: Adversarial Motion Priors for Style-Preserving Physics-Based Character Control (Peng et al., 2021)
- **链接：** <https://arxiv.org/abs/2104.02180>
- **核心贡献：** 用对抗判别器把“动作风格先验”引入 RL，显著改善 humanoid locomotion 的自然性与稳定性。
- **对 wiki 的映射：**
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [WBC vs RL](../../wiki/comparisons/wbc-vs-rl.md)

### 2) ASE: Adversarial Skill Embeddings for Hierarchical RL (Peng et al., 2022)
- **链接：** <https://arxiv.org/abs/2205.01906>
- **核心贡献：** 将技能压缩到 latent embedding，并通过对抗目标学习可组合技能空间，支持技能插值与复用。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Loco-manipulation](../../wiki/tasks/loco-manipulation.md)

### 3) Emergence of Locomotion Behaviours in Rich Environments (Heess et al., 2017)
- **链接：** <https://arxiv.org/abs/1707.02286>
- **核心贡献：** 证明了在丰富地形与奖励设计下，深度 RL 可自发涌现多样 locomotion 行为。
- **对 wiki 的映射：**
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Reward Design](../../wiki/concepts/reward-design.md)

### 4) Learning Agile Robotic Locomotion Skills by Imitating Animals (Margolis et al., 2024)
- **链接：** <https://arxiv.org/abs/2404.06818>
- **核心贡献：** 以大规模动物运动先验驱动腿足策略学习，强化跨地形与扰动鲁棒性。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] 已为 AMP / ASE / 经典 locomotion RL 论文补充可用摘要
- [~] 后续补：按“任务难度（站立/行走/奔跑/复杂地形）”重排论文脉络
- [~] 后续补：增加 PPO/SAC/TD3 在 locomotion 任务中的可复现实验对照
