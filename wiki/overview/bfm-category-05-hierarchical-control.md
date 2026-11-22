---
type: overview
tags: [bfm, behavior-foundation-model, category-hub, awesome-bfm-papers, hierarchical]
status: complete
updated: 2026-11-21
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
  - ../entities/paper-sentinel.md
  - ../methods/beyondmimic.md
  - ../entities/paper-bfm-36-leverb.md
  - ../entities/paper-bfm-37-langwbc.md
  - ../entities/paper-bfm-38-tokenhsi.md
  - ../entities/paper-bfm-39-closd.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/repos/awesome_bfm_papers.md
original:
  - ../../sources/repos/awesome_bfm_papers.md
---

# BFM 分类 05：Hierarchical control

> **分类页**：对应 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 与 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 的 **05 Hierarchical control** 分组；本页汇集该组 **8 篇** 论文的站内实体与 source 索引。总地图见 [BFM 技术地图](./bfm-41-papers-technology-map.md)。

## 一句话定义

**层次化 BFM 控制通过技能 Token、潜变量语义动作（latent verbs）、引导扩散规划器与实时跟踪器的上下解耦，解决「低频语义/视觉推理」与「高频人形全身动态平衡」之间的时间尺度与接口失配。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 层次化控制中的低层身体执行 |
| HRL | Hierarchical Reinforcement Learning | 高层技能 + 低层控制的结构 |
| LLM | Large Language Model | 部分工作作高层任务/语言接口 |
| VLA | Vision-Language-Action | 与 BFM 低层组合的上层策略 |
| WBC | Whole-Body Control | 低层跟踪与全身协调 |

## 为什么重要

- **打通 VLA 与全身动态控制的接口瓶颈：** 纯 VLA 或 LLM 直接输出高频关节位置既受限于推理延迟，又缺乏动力学平衡保障；层次 BFM 令上层专注语义/导航，下层承担 50 Hz 物理落地。
- **覆盖语言、交互与扩散规划多元范式：** 从 LangWBC 端到端 CVAE、LeVerb 双系统潜动词到 BeyondMimic、CloSD、UniPhys 与 TokenHSI，形成丰富的接口对照组。

## 核心机制与本组论文（8 篇）

语言、VLA、扩散与规划器如何通过**层次接口**（技能 token、latent action、action chunk）调用已训练好的底层身体，并由 WBC / 技能执行器承担关节级闭环？（工程分层见 [VLA 与低层控制器](../queries/vla-with-low-level-controller.md)）。核心思路是将上层低频输出（如 10 Hz latent verb、引导扩散参考帧或子任务 token）与下层 50 Hz 高频跟踪控制器以异步缓冲或采样对齐方式耦合。

**代表工作（策展）：** SENTINEL、BeyondMimic、LeVerb、LangWBC、TokenHSI、CLoSD、UniPhys、UniHSI

## 本组论文（8 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 34 | SENTINEL | [paper-sentinel.md](../entities/paper-sentinel.md) | [source](../../sources/papers/bfm_awesome_sentinel_arxiv_2511_19236.md) |
| 35 | BeyondMimic | [beyondmimic.md](../methods/beyondmimic.md) | [source](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) |
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

## 工程实践

- **对齐上下层时序分布与接口采样：** 参照 LeVERB 在训练 LeVERB-A 时从 System 2 高斯隐分布重采样（每 5 步更新并保持同步）以及 ReactiveBFM 的异步轨迹块时间集成，避免高低频直接串接引发边界跳变或分布偏移。
- **严守高/低频计算隔离：** 把 50 Hz 底层跟踪器隔离在机载实时 CPU/ONNX 进程，高层 VLM/扩散规划跑在异步 GPU/外置主机上并设缓冲阈值防抢占。

## 常见误区或局限

- **错把语言泛化瓶颈全怪上层大模型：** 若底层跟踪器本身缺乏跨接触恢复与抗扰鲁棒性，语义或规划输出再准也无法落地真机。
- **开环规划级联触发累积曝光偏差：** 仅靠一次性运动生成而不闭合本体感知反馈（如未用 scheduled prefix rollout），微小误差会在长程执行中迅速滚雪球跌倒。

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

## 推荐继续阅读

- [Behavior Foundation Model 概念页](../concepts/behavior-foundation-model.md)
- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [VLA 与低层控制器选型](../queries/vla-with-low-level-controller.md)
