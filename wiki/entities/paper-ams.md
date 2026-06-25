---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bfm, behavior-foundation-model, nvidia, hku, tsinghua]
status: complete
updated: 2026-06-25
arxiv: "2511.17373"
venue: "2025 · arXiv"
code: https://github.com/OpenDriveLab/AMS
summary: "AMS（Agility Meets Stability）：异构数据下敏捷与稳定权衡；在 RL 身体系统栈属参考跟踪层，在 BFM 谱系属 Goal-conditioned 学习。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../concepts/behavior-foundation-model.md
  - ../methods/ams.md
sources:
  - ../../sources/papers/humanoid_rl_stack_18_agility_meets_stability_versatile_humanoid_contr.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_ams_arxiv_2511_17373.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# Agility Meets Stability

**Agility Meets Stability**（AMS，arXiv:2511.17373）讨论高动态敏捷动作与稳定恢复能力在同一控制器中的权衡；MoCap/仿真/视频等异构混合是 BFM 数据常态。

> **深读页：** [ams](../methods/ams.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

AMS，全称 Agility Meets Stability，明确讨论一个矛盾：高动态敏捷动作和稳定恢复能力很难在同一个控制器里兼得。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#18/42）。
- AMS，全称 Agility Meets Stability，明确讨论一个矛盾：高动态敏捷动作和稳定恢复能力很难在同一个控制器里兼得。
- 论文利用异构数据：人类动作提供敏捷技能，合成 balance motions 提供稳定性和恢复能力。它希望一个策略既能跟踪动态动作，又能在扰动或失衡时保持稳定。
- 这篇论文让我想到一个很现实的问题：很多机器人 demo 很敏捷，但边界状态下恢复能力弱；另一类控制器很稳，但动作保守。AMS 试图把这两种能力放在同一个框架里。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 论文 | <https://arxiv.org/abs/2511.17373> |
| 项目页 | <https://opendrivelab.com/AMS/> |
| 代码 | <https://github.com/OpenDriveLab/AMS> |
| 机构 | 香港大学；NVIDIA；清华大学 |

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 18/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 09/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 核心机制（归纳）

### 1）策展导读要点

AMS，全称 Agility Meets Stability，明确讨论一个矛盾：高动态敏捷动作和稳定恢复能力很难在同一个控制器里兼得。

### 2）策展导读要点

论文利用异构数据：人类动作提供敏捷技能，合成 balance motions 提供稳定性和恢复能力。它希望一个策略既能跟踪动态动作，又能在扰动或失衡时保持稳定。

### 3）策展导读要点

这篇论文让我想到一个很现实的问题：很多机器人 demo 很敏捷，但边界状态下恢复能力弱；另一类控制器很稳，但动作保守。AMS 试图把这两种能力放在同一个框架里。

### 4）策展导读要点

我的判断**真正可部署的人形机器人不能只做敏捷动作，也不能只做保守稳定，它必须在二者之间动态切换。**

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 方法页：[ams.md](../methods/ams.md)
- RL 身体系统栈：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- BFM 技术地图：[bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)

## 参考来源

- [humanoid_rl_stack_18_agility_meets_stability_versatile_humanoid_contr.md](../../sources/papers/humanoid_rl_stack_18_agility_meets_stability_versatile_humanoid_contr.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 42 篇总表
- [bfm_awesome_ams_arxiv_2511_17373.md](../../sources/papers/bfm_awesome_ams_arxiv_2511_17373.md) — awesome-bfm 策展摘录
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 41+10 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — RL 运动控制微信公众号编译导读
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) — BFM 41 篇微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2511.17373>

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 完整列表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 综述
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
