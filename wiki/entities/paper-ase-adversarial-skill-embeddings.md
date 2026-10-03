---
type: entity
tags: [paper, motion-prior, skill-learning, adversarial-imitation, character-control]
status: complete
updated: 2026-10-03
arxiv: "2205.01906"
related:
  - ./paper-bfm-zero.md
  - ./paper-behavior-foundation-model-humanoid.md
  - ../methods/amp-reward.md
sources:
  - ../../sources/papers/ase.md
  - ../../sources/papers/bfm_awesome_ase_arxiv_2205_01906.md
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "ASE（SIGGRAPH 2022）：学习可由潜变量选择的大规模可复用对抗技能嵌入，为上层调用运动能力提供接口。"
---

# ASE：可复用对抗技能嵌入

**ASE**（*ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters*，[arXiv:2205.01906](https://arxiv.org/abs/2205.01906)，SIGGRAPH 2022）从动作数据学习一组可由潜变量调用的技能表示，使高层控制器可以选择不同身体行为。

## 一句话理解

把多样动作压成可以选择和复用的技能，而不是只训练一套固定步态。

## 方法要点

- 通过无监督强化学习与对抗目标学习动作技能嵌入。
- 潜在技能变量为高层提供行为选择接口。
- 在 Day 2 的运动先验脉络中，ASE 与 AMP 的风格约束、BFM-Zero 的目标提示式调用构成不同层次的技能复用方法。

## 与相邻工作

- [AMP](./paper-amp-survey-01-amp.md)主要学习示范动作风格先验。
- [BFM-Zero](./paper-bfm-zero.md)研究通过提示调用预训练行为。
- 此页对应 ASE 论文，不与缩写或综述页合并。

## 来源

- [ASE 项目页](https://xbpeng.com/projects/ASE/index.html)
- [arXiv:2205.01906](https://arxiv.org/abs/2205.01906)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
