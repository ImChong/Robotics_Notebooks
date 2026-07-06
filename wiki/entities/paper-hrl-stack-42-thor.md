---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, baai, bit]
status: complete
updated: 2026-07-06
venue: curated
summary: "这篇中文文件名是“北京理工：雷神”，论文题目对应 Human-Level Whole-Body Reactions for Intense Contact-Rich Environments。它关注的是强接触环境中的全身反应。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-09-compliance-contact.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_42_thor_towards_human_level_whole_body_reactions_fo.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Thor

**Thor** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 42/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

## 一句话定义

这篇中文文件名是“北京理工：雷神”，论文题目对应 Human-Level Whole-Body Reactions for Intense Contact-Rich Environments。它关注的是强接触环境中的全身反应。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#42/42）。
- 这篇中文文件名是“北京理工：雷神”，论文题目对应 Human-Level Whole-Body Reactions for Intense Contact-Rich Environments。它关注的是强接触环境中的全身反应。
- 机器人在服务、工业、救援场景中可能需要持续和环境发生强接触，比如拉、推、支撑、撞击、搬运。传统方法如果只关注末端或下肢，很难产生类似人的全身协调反应。
- 论文设计 force-adaptive torso-related reward，并提出 Thor RL architecture，把上肢和下肢解耦但共享全身信息。它希望机器人在强交互任务中通过躯干、上肢、下肢协同产生更大、更稳定的作用力。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 42/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | 北京理工大学；北京智源人工智能研究院 |
| 出处 | curated |
| 链接 | <https://baai-aether.github.io/baai-thor/> |

## 核心机制（归纳）

### 1）策展导读要点

这篇中文文件名是“北京理工：雷神”，论文题目对应 Human-Level Whole-Body Reactions for Intense Contact-Rich Environments。它关注的是强接触环境中的全身反应。

### 2）策展导读要点

机器人在服务、工业、救援场景中可能需要持续和环境发生强接触，比如拉、推、支撑、撞击、搬运。传统方法如果只关注末端或下肢，很难产生类似人的全身协调反应。

### 3）策展导读要点

论文设计 force-adaptive torso-related reward，并提出 Thor RL architecture，把上肢和下肢解耦但共享全身信息。它希望机器人在强交互任务中通过躯干、上肢、下肢协同产生更大、更稳定的作用力。

### 4）策展导读要点

我把它放在 GentleHumanoid 和 CHIP 之后看：GentleHumanoid 强调安全柔顺，CHIP 强调可调末端刚度，Thor 强调强接触下的全身发力。

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_42_thor_towards_human_level_whole_body_reactions_fo.md](../../sources/papers/humanoid_rl_stack_42_thor_towards_human_level_whole_body_reactions_fo.md)

## 参考来源

- [humanoid_rl_stack_42_thor_towards_human_level_whole_body_reactions_fo.md](../../sources/papers/humanoid_rl_stack_42_thor_towards_human_level_whole_body_reactions_fo.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
