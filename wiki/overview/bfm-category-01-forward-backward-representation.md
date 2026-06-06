---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, forward-backward]
status: complete
updated: 2026-05-27
summary: "具身智能研究室 BFM 41 篇专题 · 01 Forward-backward 表征（6 篇）— 多任务能否压进可调用的身体潜空间（latent prompt / FB 嵌入），而非每换一个任务就重训全身策略？"
related:
  - ./bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ./bfm-category-02-goal-conditioned-learning.md
  - ./bfm-category-03-intrinsic-reward-pretraining.md
  - ./bfm-category-04-adaptation.md
  - ./bfm-category-05-hierarchical-control.md
  - ../entities/paper-bfm-01-bfm-zero.md
  - ../entities/paper-bfm-02-metamotivo.md
  - ../entities/paper-bfm-03-fb-aw.md
  - ../entities/paper-bfm-04-fast-imitation-bfm.md
  - ../entities/paper-bfm-05-learning-one-representation.md
  - ../entities/paper-bfm-06-successor-states.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 01：Forward-backward 表征

> **图谱分类节点**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **01 Forward-backward 表征** 分组；本页汇集该组 **6 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 核心问题（公众号分类）

多任务能否压进**可调用的身体潜空间**（latent prompt / FB 嵌入），而非每换一个任务就重训全身策略？

**代表工作（策展）：** BFM-Zero、MetaMotivo、FB-AW、Fast Imitation、Learning One Representation、Successor States

## 本组论文（6 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | BFM-Zero | [paper-bfm-01-bfm-zero.md](../entities/paper-bfm-01-bfm-zero.md) | [source](../../sources/papers/bfm_awesome_bfm_zero_arxiv_2511_04131.md) |
| 02 | Zero-shot Whole-body Humanoid Control via Behavioral Foundation Models | [paper-bfm-02-metamotivo.md](../entities/paper-bfm-02-metamotivo.md) | [source](../../sources/papers/bfm_awesome_metamotivo_arxiv_2504_11054.md) |
| 03 | Finer Behavioral Foundation Models via Auto-regressive Features and Advantage Weighting | [paper-bfm-03-fb-aw.md](../entities/paper-bfm-03-fb-aw.md) | [source](../../sources/papers/bfm_awesome_fb_aw_arxiv_2412_04368.md) |
| 04 | Fast Imitation via Behavior Foundation Models | [paper-bfm-04-fast-imitation-bfm.md](../entities/paper-bfm-04-fast-imitation-bfm.md) | [source](../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md) |
| 05 | Learning One Representation to Optimize All Rewards | [paper-bfm-05-learning-one-representation.md](../entities/paper-bfm-05-learning-one-representation.md) | [source](../../sources/papers/bfm_awesome_learning_one_representation_neurips_2021.md) |
| 06 | Learning Successor States and Goal-Dependent Values | [paper-bfm-06-successor-states.md](../entities/paper-bfm-06-successor-states.md) | [source](../../sources/papers/bfm_awesome_successor_states_arxiv_2101_07123.md) |

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 01 Forward-backward 表征 |
| 篇数 | 6/41 |
| 概念对照 | [Behavior Foundation Model](../concepts/behavior-foundation-model.md) |
| 姊妹分类 | 见 [BFM 技术地图 · 五类问题](./bfm-41-papers-technology-map.md#流程总览五类问题--身体-api) |

## 关联页面

- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)

## 参考来源

- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（TPAMI 2025）
