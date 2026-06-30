---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, stanford, berkeley]
status: complete
updated: 2026-06-30
arxiv: "2508.08241"
venue: "2025 · arXiv"
code: https://github.com/HybridRobotics/whole_body_tracking
summary: "BeyondMimic：从 motion tracking 到 guided diffusion 的通用人形控制；在 RL 身体系统栈属参考跟踪层，在 BFM 谱系属 hierarchical control。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-05-hierarchical-control.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/beyondmimic.md
sources:
  - ../../sources/papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# BeyondMimic

**BeyondMimic**（*From Motion Tracking to Versatile Humanoid Control via Guided Diffusion*，arXiv:2508.08241）将 guided diffusion 引入全身人形控制。

> **深读页：** [beyondmimic](../methods/beyondmimic.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#15/42）。
- BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。
- 普通 motion tracking 通常假设参考轨迹已经给定，控制器只需要尽量追踪。但真实任务里，参考动作可能缺局部细节，可能和当前环境不匹配，也可能需要根据任务目标在线调整。
- BeyondMimic 的思路是引入 guided diffusion，让策略不只是被动追踪参考，而是能在约束和目标引导下生成、修正更合适的动作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2508.08241> |
| 代码 | <https://github.com/HybridRobotics/whole_body_tracking> |
| 机构 | 伯克利；斯坦福大学 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 15/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 35/41 |
| 分组 | 05 Hierarchical control |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 核心机制（归纳）

### 1）策展导读要点

BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。

### 2）策展导读要点

普通 motion tracking 通常假设参考轨迹已经给定，控制器只需要尽量追踪。但真实任务里，参考动作可能缺局部细节，可能和当前环境不匹配，也可能需要根据任务目标在线调整。

### 3）策展导读要点

BeyondMimic 的思路是引入 guided diffusion，让策略不只是被动追踪参考，而是能在约束和目标引导下生成、修正更合适的动作。

### 4）策展导读要点

这篇论文的价值在于，它把“模仿”往“可控生成”推进了一步。DeepMimic 那条线解决的是“如何跟着人类动作学”；OmniTrack 和 RGMT 解决的是“参考动作怎样更物理、更鲁棒”；BeyondMimic 则进一步问：如果参考本身不够，能不能让模型在物理约束下补出更可执行的行为？

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 方法深读：[beyondmimic.md](../methods/beyondmimic.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)

## 参考来源

- [humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md](../../sources/papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_beyondmimic_arxiv_2508_08241.md](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2508.08241>

## 推荐继续阅读

- [机器人论文阅读笔记：BeyondMimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/01_Foundational_RL/BeyondMimic/BeyondMimic.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
