# Locomotion RL

聚焦人形/腿足机器人 locomotion 中的强化学习论文。

## 关注问题

- 如何让机器人走得更稳？
- 如何提升速度与能效？
- 如何提高扰动恢复能力？
- 如何增强 sim2real 可迁移性？

## 代表性论文

### 基础算法

- **PPO** (Schulman et al., 2017) — Humanoid locomotion RL 最常用基线，[arXiv](https://arxiv.org/abs/1707.06347)
- **SAC** (Haarnoja et al., 2018) — 连续控制常用 off-policy 算法，[arXiv](https://arxiv.org/abs/1801.01290)

### 人形/足式 Locomotion

- **AMP** — Adversarial Motion Priors (Peng et al., 2021)，用对抗方式嵌入技能，[Code](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_motion_priors)
- **ASE** — Adversarial Skill Embeddings (Peng et al., 2022)，对抗训练 latent 技能先验，[Code](https://github.com/google-deepmind/ase)
- **RMA** — Rapid Motor Adaptation (Kumar et al., 2021)，四足快速自适应，[arXiv](https://arxiv.org/abs/2107.04034)

### 人形全身控制

- **CALM** — Conditioned Latent Action Models (Tessler et al., 2023)，latent 空间控制，[Code](https://github.com/irlet/calm)
- **MimicKit** — 模仿学习 + 技能迁移 framework，[Project Page](https://motion.stanford.edu/research/mimickit)

## 关联页面

- [[wiki/concepts/sim2real]] — Sim2Real 是 locomotion RL 的核心挑战
- [[wiki/concepts/domain-randomization]] — DR 是当前主流 sim2real 方法
- [[wiki/concepts/locomotion]] — locomotion 任务层
