---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, stanford]
status: complete
updated: 2026-07-06
venue: curated
summary: "GentleHumanoid 关注 upper-body compliance for contact-rich human and object interaction。它的目标是让人形机器人在握手、拥抱、辅助坐站、气球操作等任务里保持安全、自然的接触。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-09-compliance-contact.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/gentlehumanoid-motion-tracking.md
sources:
  - ../../sources/papers/humanoid_rl_stack_37_gentlehumanoid_learning_upper_body_compliance_fo.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# GentleHumanoid

**GentleHumanoid** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 37/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

> **深读页：** [gentlehumanoid-motion-tracking](../methods/gentlehumanoid-motion-tracking.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

GentleHumanoid 关注 upper-body compliance for contact-rich human and object interaction。它的目标是让人形机器人在握手、拥抱、辅助坐站、气球操作等任务里保持安全、自然的接触。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#37/42）。
- GentleHumanoid 关注 upper-body compliance for contact-rich human and object interaction。它的目标是让人形机器人在握手、拥抱、辅助坐站、气球操作等任务里保持安全、自然的接触。
- 它的核心设计是 reference dynamics：把上肢关节建模为 spring-damper 系统，让参考轨迹随着接触力变化。策略不再跟踪固定目标，而是跟踪会根据接触变形的 reference。
- 这和外部套一个 impedance controller 不同。GentleHumanoid 把顺应性放进 RL 的奖励和参考状态里，让策略在训练阶段就学会接触。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 37/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | 斯坦福大学 |
| 出处 | curated |
| 链接 | <https://gentle-humanoid.axell.top> |

## 核心机制（归纳）

### 1）策展导读要点

GentleHumanoid 关注 upper-body compliance for contact-rich human and object interaction。它的目标是让人形机器人在握手、拥抱、辅助坐站、气球操作等任务里保持安全、自然的接触。

### 2）策展导读要点

它的核心设计是 reference dynamics：把上肢关节建模为 spring-damper 系统，让参考轨迹随着接触力变化。策略不再跟踪固定目标，而是跟踪会根据接触变形的 reference。

### 3）策展导读要点

这和外部套一个 impedance controller 不同。GentleHumanoid 把顺应性放进 RL 的奖励和参考状态里，让策略在训练阶段就学会接触。

### 4）策展导读要点

论文还加入安全力阈值机制，避免接触力无限增长。实验覆盖外力扰动、拥抱人体模特、坐站辅助和气球操作。

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_37_gentlehumanoid_learning_upper_body_compliance_fo.md](../../sources/papers/humanoid_rl_stack_37_gentlehumanoid_learning_upper_body_compliance_fo.md)

## 参考来源

- [humanoid_rl_stack_37_gentlehumanoid_learning_upper_body_compliance_fo.md](../../sources/papers/humanoid_rl_stack_37_gentlehumanoid_learning_upper_body_compliance_fo.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
