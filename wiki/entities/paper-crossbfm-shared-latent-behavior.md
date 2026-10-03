---
type: entity
tags: [paper, humanoid, behavior-foundation-model, cross-embodiment, distillation]
status: complete
updated: 2026-10-03
arxiv: "2609.38087"
related:
  - ./paper-behavior-foundation-model-humanoid.md
  - ./paper-any2any-cross-embodiment-wbt.md
  - ../concepts/motion-retargeting.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "CrossBFM（arXiv:2609.38087）：用跨本体对应动作蒸馏共享潜在行为空间，同时保留各机器人专用的低层跟踪器。"
---

# CrossBFM：跨人形本体蒸馏共享行为空间

**CrossBFM**（*Distilling a Shared Latent Behavior Space Across Humanoid Embodiments*，[arXiv:2609.38087](https://arxiv.org/abs/2609.38087)，[项目页](https://dotandung.github.io/crossbfm/)）研究如何让同一个行为提示在不同人形机器人的潜在动作空间中具有相近含义。

## 一句话理解

对齐不同本体的动作表示，让共享行为提示由每台机器人自己的跟踪控制器落实。

## 方法要点

- 利用跨本体对应动作蒸馏共享行为编码器。
- 共享的是高层行为表示；各机器人仍保留各自的身体控制器。
- 该设计将跨机器人复用放在行为接口上，而不是直接复制关节命令。

## 与相邻工作

[BFM](./paper-behavior-foundation-model-humanoid.md)讨论人形行为基础模型；CrossBFM进一步处理跨本体表示对齐。本页对应独立论文，不与一般跨本体跟踪工作合并。

## 来源

- [arXiv:2609.38087](https://arxiv.org/abs/2609.38087)
- [项目页](https://dotandung.github.io/crossbfm/)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)
