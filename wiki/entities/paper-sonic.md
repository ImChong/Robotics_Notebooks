---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, nvidia]
status: complete
updated: 2026-06-26
arxiv: "2511.07820"
venue: "2025 · arXiv"
summary: "SONIC：规模化运动跟踪人形全身控制；在 RL 身体系统栈属参考跟踪层，在 BFM 谱系强调 goal-conditioned 与运控基座覆盖面。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/sonic-motion-tracking.md
sources:
  - ../../sources/papers/humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_sonic_arxiv_2511_07820.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# SONIC

**SONIC**（*Supersizing Motion Tracking for Natural Humanoid Whole-Body Control*，arXiv:2511.07820）把 humanoid whole-body motion tracker 当作可扩展的基础模型来研究。

> **深读页：** [sonic-motion-tracking](../methods/sonic-motion-tracking.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#17/42）。
- SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。
- 这篇论文和传统 motion tracking 工作的区别在于，它不只问“某个策略能不能跟某些动作”，而是问：
- 它还讨论下游任务、交互式运动控制，以及 motion 和 VLA 表示的迁移价值。这意味着 motion tracker 不再只是一个执行模块，而可能成为上层任务模型的底层表征。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2511.07820> |
| 项目页 | <https://nvlabs.github.io/SONIC/> |
| 机构 | NVIDIA |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 17/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 07/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 核心机制（归纳）

### 1）策展导读要点

SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。

### 2）策展导读要点

这篇论文和传统 motion tracking 工作的区别在于，它不只问“某个策略能不能跟某些动作”，而是问：

### 3）策展导读要点

它还讨论下游任务、交互式运动控制，以及 motion 和 VLA 表示的迁移价值。这意味着 motion tracker 不再只是一个执行模块，而可能成为上层任务模型的底层表征。

### 4）策展导读要点

SONIC 的方向很重要，但也要谨慎：运动控制的 scaling 不会和语言模型完全一样，因为机器人还受硬件、物理和实时闭环约束。参数变大不一定能直接解决接触和安全问题。

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 方法深读：[sonic-motion-tracking.md](../methods/sonic-motion-tracking.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- BFM 概念：[behavior-foundation-model.md](../concepts/behavior-foundation-model.md)

## 参考来源

- [humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md](../../sources/papers/humanoid_rl_stack_17_sonic_supersizing_motion_tracking_for_natural_hu.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_sonic_arxiv_2511_07820.md](../../sources/papers/bfm_awesome_sonic_arxiv_2511_07820.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2511.07820>

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
