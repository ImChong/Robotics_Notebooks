---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, adaptation]
status: complete
updated: 2026-05-27
summary: "具身智能研究室 BFM 41 篇专题 · 04 Adaptation（3 篇）— 预训练 BFM 如何以低成本适配新任务、新动力学或新机体（样本与工程摩擦）？"
related:
  - ./bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ./bfm-category-01-forward-backward-representation.md
  - ./bfm-category-02-goal-conditioned-learning.md
  - ./bfm-category-03-intrinsic-reward-pretraining.md
  - ./bfm-category-05-hierarchical-control.md
  - ../entities/paper-bfm-31-task-tokens.md
  - ../entities/paper-bfm-32-unseen-dynamics.md
  - ../entities/paper-bfm-33-fast-adaptation-bfm.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 04：Adaptation

> **图谱分类节点**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **04 Adaptation** 分组；本页汇集该组 **3 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 核心问题（公众号分类）

预训练 BFM 如何以**低成本**适配新任务、新动力学或新机体（样本与工程摩擦）？

**代表工作（策展）：** Task Tokens、Unseen Dynamics、Fast Adaptation

## 本组论文（3 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 31 | Task Tokens | [paper-bfm-31-task-tokens.md](../entities/paper-bfm-31-task-tokens.md) | [source](../../sources/papers/bfm_awesome_task_tokens_arxiv_2503_22886.md) |
| 32 | Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics | [paper-bfm-32-unseen-dynamics.md](../entities/paper-bfm-32-unseen-dynamics.md) | [source](../../sources/papers/bfm_awesome_unseen_dynamics_arxiv_2505_13150.md) |
| 33 | Fast Adaptation With Behavioral Foundation Models | [paper-bfm-33-fast-adaptation-bfm.md](../entities/paper-bfm-33-fast-adaptation-bfm.md) | [source](../../sources/papers/bfm_awesome_fast_adaptation_bfm_corl_2025.md) |

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 04 Adaptation |
| 篇数 | 3/41 |
| 概念对照 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 姊妹分类 | 见 [BFM 技术地图 · 五类问题](./bfm-41-papers-technology-map.md#流程总览五类问题--身体-api) |

## 关联页面

- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 参考来源

- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（TPAMI 2025）
