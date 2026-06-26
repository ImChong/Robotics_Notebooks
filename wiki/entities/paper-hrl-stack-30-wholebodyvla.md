---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, opendrivelab, agibot, hku, fudan]
status: complete
updated: 2026-06-26
venue: curated
summary: "WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_30_wholebodyvla_towards_unified_latent_vla_for_whol.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# WholeBodyVLA

**WholeBodyVLA** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 30/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

## 一句话定义

WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#30/42）。
- WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。
- 它的关键组件包括 Unified Latent Learning 和 LMO。前者把 action-free 视频转成 latent action token，后者则是面向 loco-manipulation 的底层 RL 策略。
- 这篇论文里我最认同的一点是：行走不是为了追踪速度，而是为了到达适合操作的位置，并让身体稳定下来。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 30/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | 复旦大学；OpenDriveLab & 香港大学 MMLab；智元机器人；SII |
| 出处 | curated |
| 链接 | <https://opendrivelab.com/WholeBodyVLA> |

## 核心机制（归纳）

### 1）策展导读要点

WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。

### 2）策展导读要点

它的关键组件包括 Unified Latent Learning 和 LMO。前者把 action-free 视频转成 latent action token，后者则是面向 loco-manipulation 的底层 RL 策略。

### 3）策展导读要点

这篇论文里我最认同的一点是：行走不是为了追踪速度，而是为了到达适合操作的位置，并让身体稳定下来。

### 4）策展导读要点

如果底层 locomotion 只会速度跟踪，上层 VLA 就要自己学习“如何走到适合抓取的位置”。这会让任务变得很难。WholeBodyVLA 通过面向操作的 locomotion controller，把行走重新定义成操作准备动作。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_30_wholebodyvla_towards_unified_latent_vla_for_whol.md](../../sources/papers/humanoid_rl_stack_30_wholebodyvla_towards_unified_latent_vla_for_whol.md)

## 参考来源

- [humanoid_rl_stack_30_wholebodyvla_towards_unified_latent_vla_for_whol.md](../../sources/papers/humanoid_rl_stack_30_wholebodyvla_towards_unified_latent_vla_for_whol.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
