---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, hierarchical]
status: complete
updated: 2026-05-27
summary: "具身智能研究室 BFM 41 篇专题 · 05 Hierarchical control（8 篇）— 语言、VLA、扩散与规划器如何通过层次接口（技能 token、action chunk）调用已训练好的底层身体？"
related:
  - ./bfm-41-papers-technology-map.md
  - ../concepts/behavior-foundation-model.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ./bfm-category-01-forward-backward-representation.md
  - ./bfm-category-02-goal-conditioned-learning.md
  - ./bfm-category-03-intrinsic-reward-pretraining.md
  - ./bfm-category-04-adaptation.md
  - ../entities/paper-bfm-34-sentinel.md
  - ../entities/paper-bfm-35-beyondmimic.md
  - ../entities/paper-bfm-36-leverb.md
  - ../entities/paper-bfm-37-langwbc.md
  - ../entities/paper-bfm-38-tokenhsi.md
  - ../entities/paper-bfm-39-closd.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 05：Hierarchical control

> **图谱分类节点**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **05 Hierarchical control** 分组；本页汇集该组 **8 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 核心问题（公众号分类）

语言、VLA、扩散与规划器如何通过**层次接口**（技能 token、latent action、action chunk）调用已训练好的底层身体，并由 WBC / 技能执行器承担关节级闭环？（工程分层见 [VLA 与低层控制器](../queries/vla-with-low-level-controller.md)）

**代表工作（策展）：** SENTINEL、BeyondMimic、LeVerb、LangWBC、TokenHSI、CLoSD、UniPhys、UniHSI

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 层次化控制中的低层身体执行 |
| HRL | Hierarchical Reinforcement Learning | 高层技能 + 低层控制的结构 |
| LLM | Large Language Model | 部分工作作高层任务/语言接口 |
| VLA | Vision-Language-Action | 与 BFM 低层组合的上层策略 |
| WBC | Whole-Body Control | 低层跟踪与全身协调 |

## 本组论文（8 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 34 | SENTINEL | [paper-bfm-34-sentinel.md](../entities/paper-bfm-34-sentinel.md) | [source](../../sources/papers/bfm_awesome_sentinel_arxiv_2511_19236.md) |
| 35 | BeyondMimic | [paper-bfm-35-beyondmimic.md](../entities/paper-bfm-35-beyondmimic.md) | [source](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) |
| 36 | LeVerb | [paper-bfm-36-leverb.md](../entities/paper-bfm-36-leverb.md) | [source](../../sources/papers/bfm_awesome_leverb_arxiv_2506_13751.md) |
| 37 | LangWBC | [paper-bfm-37-langwbc.md](../entities/paper-bfm-37-langwbc.md) | [source](../../sources/papers/bfm_awesome_langwbc_arxiv_2504_21738.md) |
| 38 | Tokenhsi | [paper-bfm-38-tokenhsi.md](../entities/paper-bfm-38-tokenhsi.md) | [source](../../sources/papers/bfm_awesome_tokenhsi_arxiv_2503_19901.md) |
| 39 | CloSD | [paper-bfm-39-closd.md](../entities/paper-bfm-39-closd.md) | [source](../../sources/papers/bfm_awesome_closd_arxiv_2410_03441.md) |
| 40 | UniPhys | [paper-bfm-40-uniphys.md](../entities/paper-bfm-40-uniphys.md) | [source](../../sources/papers/bfm_awesome_uniphys_arxiv_2504_12540.md) |
| 41 | Unified Human-Scene Interaction via Prompted Chain-of-Contacts | [paper-bfm-41-unihsi.md](../entities/paper-bfm-41-unihsi.md) | [source](../../sources/papers/bfm_awesome_unihsi_arxiv_2309_07918.md) |

## 在 BFM taxonomy 中的位置

| 字段 | 内容 |
|------|------|
| 分组 | 05 Hierarchical control |
| 篇数 | 8/41 |
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
