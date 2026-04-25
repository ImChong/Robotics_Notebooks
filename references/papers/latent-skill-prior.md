# Latent Skill Prior

聚焦如何将运动技能压缩到隐空间（Latent Space），并通过先验（Prior）引导强化学习生成既稳定又自然的行为。

## 关注问题

- 如何从大规模动捕数据中提炼通用的运动先验？
- 如何在隐空间内实现不同技能的平滑插值与组合？
- 如何解决“既要像人/动物一样自然，又要完成特定任务”的多目标冲突？
- 如何将预训练的技能先验迁移到不同形态或物理特性的机器人上？

## 代表性论文

### 核心方法论

- **AMP** — *Adversarial Motion Priors for Style-Preserving Physics-Based Character Control* (Peng et al., 2021). 用对抗判别器将动作风格引入 RL。[arXiv](https://arxiv.org/abs/2104.02180)
- **ASE** — *Adversarial Skill Embeddings for Hierarchical RL* (Peng et al., 2022). 将技能压缩到 latent embedding，学习可组合的技能空间。[arXiv](https://arxiv.org/abs/2205.01906)

### 技能条件化与组合

- **CALM** — *Conditional Latent Action Models* (Tessler et al., 2023). 通过条件化的隐空间实现更精细的行为控制。[Project Page](https://irlet.github.io/calm/)
- **MimicKit** — (Peng et al.). 一套完整的模仿学习与技能迁移框架，支持大规模数据集的高效利用。[Code](https://github.com/xbpeng/MimicKit)

### 应用与演进

- **ASE on Humanoids** — 将隐空间技能先验应用于双足人形，解决全身协调控制难题。

## 关联页面

- [Locomotion RL (Reference)](./locomotion-rl.md)
- [Reinforcement Learning (Method)](../../wiki/methods/reinforcement-learning.md)
- [Imitation Learning (Method)](../../wiki/methods/imitation-learning.md)
- [Foundation Policy (Concept)](../../wiki/concepts/foundation-policy.md)
