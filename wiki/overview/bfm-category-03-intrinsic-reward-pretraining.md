---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, intrinsic-reward]
status: complete
updated: 2026-05-27
summary: "具身智能研究室 BFM 41 篇专题 · 03 Intrinsic reward 预训练（5 篇）— 在尚无明确下游任务时，身体应先通过内在奖励积累何种可迁移探索经验？"
related:
  - ./bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ./bfm-category-01-forward-backward-representation.md
  - ./bfm-category-02-goal-conditioned-learning.md
  - ./bfm-category-04-adaptation.md
  - ./bfm-category-05-hierarchical-control.md
  - ../entities/paper-bfm-26-aps.md
  - ../entities/paper-bfm-27-proto-rl.md
  - ../entities/paper-bfm-28-re3.md
  - ../entities/paper-bfm-29-rnd.md
  - ../entities/paper-bfm-30-diayn.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 03：Intrinsic reward 预训练

> **图谱分类节点**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **03 Intrinsic reward 预训练** 分组；本页汇集该组 **5 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 核心问题（公众号分类）

在**尚无明确下游任务**时，身体应先通过内在奖励积累何种**可迁移探索经验**？

**代表工作（策展）：** APS、Proto-RL、RE3、RND、DIAYN

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 26 | Active Pretraining with Successor Features | [paper-bfm-26-aps.md](../entities/paper-bfm-26-aps.md) | [source](../../sources/papers/bfm_awesome_aps_icml_2021.md) |
| 27 | Reinforcement Learning with Prototypical Representations | [paper-bfm-27-proto-rl.md](../entities/paper-bfm-27-proto-rl.md) | [source](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md) |
| 28 | State Entropy Maximization with Random Encoders for Efficient Exploration | [paper-bfm-28-re3.md](../entities/paper-bfm-28-re3.md) | [source](../../sources/papers/bfm_awesome_re3_icml_2020.md) |
| 29 | Exploration by Random Network Distillation | [paper-bfm-29-rnd.md](../entities/paper-bfm-29-rnd.md) | [source](../../sources/papers/bfm_awesome_rnd_iclr_2019.md) |
| 30 | Diversity is All You Need | [paper-bfm-30-diayn.md](../entities/paper-bfm-30-diayn.md) | [source](../../sources/papers/bfm_awesome_diayn_iclr_2018.md) |

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 03 Intrinsic reward 预训练 |
| 篇数 | 5/41 |
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
| RND | Random Network Distillation | 用预测误差作内在探索奖励的无监督手段 |
| DIAYN | Diversity Is All You Need | 以内在奖励发现多样技能的无监督预训练代表 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 参考来源

- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — 微信公众号编译（<https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g>）
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（TPAMI 2025）
