---
type: reference_page
status: complete
topic: [motion-imitation, reinforcement-learning]
summary: "隐空间技能先验（AMP / ASE / CALM / MimicKit）论文与代码导航：从对抗动作风格先验到可组合隐空间技能控制与人形迁移。"
sources:
  - https://arxiv.org/abs/2104.02180
  - https://arxiv.org/abs/2205.01906
  - https://github.com/xbpeng/MimicKit
  - ../../sources/papers/amp.md
  - ../../sources/papers/ase.md
updated: 2026-11-24
---

# Latent Skill Prior（隐空间技能先验论文导航）

**首屏导读**：本页整理将动作捕捉与示范数据压缩进**隐空间（Latent Space）**、并通过先验引导强化学习生成自然且可控行为的代表论文（AMP → ASE → CALM / MimicKit）；面向已了解强化学习基本概念、想快速梳理技能先验演进谱系的读者。

## 按问题找论文

| 想回答的问题 | 先读 |
|---|---|
| 怎样把动捕风格直接写成 RL 的对抗奖励？ | **AMP**（[arXiv:2104.02180](https://arxiv.org/abs/2104.02180)）· [AMP 实体](../../wiki/entities/paper-amp-survey-01-amp.md) · [AMP 奖励方法](../../wiki/methods/amp-reward.md) |
| 如何从无标签动捕数据学出可复用的隐空间技能先验？ | **ASE**（[arXiv:2205.01906](https://arxiv.org/abs/2205.01906)）· [ASE 归档](../../sources/papers/ase.md) |
| 如何在隐空间上做语言或目标条件化的精细控制？ | **CALM** · [CALM 实体](../../wiki/entities/paper-bfm-19-calm.md) |
| 有没有统一复现 AMP / ASE / 模仿学习的开源代码底座？ | [MimicKit (Entity)](../../wiki/entities/mimickit.md) |

## 选读路径 / Reading Path

```mermaid
flowchart LR
    MOCAP["大规模 MoCap 动作数据"] --> AMP["AMP：对抗判别器提供风格奖励"]
    AMP --> ASE["ASE：预训练低层策略 + 球面隐技能先验"]
    ASE --> CALM["CALM：条件化隐空间动作编码与复用"]
    CALM --> DOWN["下游任务规划 / 高层 RL / 人形部署"]
    MK["MimicKit / BFM 路线"] -. 统一训练与迁移工程框架 .-> ASE
    MK -.-> CALM
```

## 关注问题

- 如何从大规模动捕数据中提炼通用的运动先验？
- 如何在隐空间内实现不同技能的平滑插值与组合？
- 如何解决“既要像人/动物一样自然，又要完成特定任务”的多目标冲突？
- 如何将预训练的技能先验迁移到不同形态或物理特性的机器人上？

## 代表性论文

### 核心方法论

- **AMP** — *Adversarial Motion Priors for Style-Preserving Physics-Based Character Control* (Peng et al., 2021). 用对抗判别器将动作风格引入 RL；[arXiv](https://arxiv.org/abs/2104.02180)，实体页见 [AMP](../../wiki/entities/paper-amp-survey-01-amp.md)。
- **ASE** — *Adversarial Skill Embeddings for Hierarchical RL* (Peng et al., 2022). 将技能压缩到 latent embedding，学习可组合的技能空间；[arXiv](https://arxiv.org/abs/2205.01906)。

### 技能条件化与组合

- **CALM** — *Conditional Latent Action Models* (Tessler et al., 2023). 通过条件化的隐空间实现更精细的行为控制；[Project Page](https://irlet.github.io/calm/)，实体页见 [CALM](../../wiki/entities/paper-bfm-19-calm.md)。
- **MimicKit** — (Peng et al.). 一套完整的模仿学习与技能迁移框架，支持大规模数据集的高效利用；[Code](https://github.com/xbpeng/MimicKit)，实体页见 [MimicKit](../../wiki/entities/mimickit.md)。

### 应用与演进

- **ASE on Humanoids** — 将隐空间技能先验应用于双足人形，解决全身协调控制难题；进一步脉络见 [行为基础模型（BFM）纵深路线](../../roadmap/depth-bfm.md)。

## 参考来源与收录边界

- [AMP 原文入口（arXiv:2104.02180）](https://arxiv.org/abs/2104.02180) 与 [sources/papers/amp.md](../../sources/papers/amp.md)
- [ASE 原文入口（arXiv:2205.01906）](https://arxiv.org/abs/2205.01906) 与 [sources/papers/ase.md](../../sources/papers/ase.md)
- [MimicKit 官方仓库](https://github.com/xbpeng/MimicKit)
- 本页聚焦技能先验谱系索引；单篇实验与代码入口由对应 `wiki/entities/` 节点负责核验。

## 关联页面

- [Locomotion RL (Reference)](./locomotion-rl.md)
- [Imitation Learning (Reference)](./imitation-learning.md)
- [AMP 奖励方法 (Method)](../../wiki/methods/amp-reward.md)
- [MimicKit (Entity)](../../wiki/entities/mimickit.md)
- [Reinforcement Learning (Method)](../../wiki/methods/reinforcement-learning.md)
- [Imitation Learning (Method)](../../wiki/methods/imitation-learning.md)
- [Foundation Policy (Concept)](../../wiki/concepts/foundation-policy.md)
- [行为基础模型（BFM）纵深路线](../../roadmap/depth-bfm.md)
