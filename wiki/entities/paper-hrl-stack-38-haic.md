---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, hkust-gz, xiaomi-robotics, eth, hkust, tsinghua]
status: complete
updated: 2026-07-16
venue: curated
summary: "HAIC 关注 Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。它处理的是 underactuated objects，也就是对象本身有独立动力学和非完整约束，不是被机器人末端完全控制的刚性物体。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# HAIC

**HAIC** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 38/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

## 一句话定义

HAIC 关注 Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。它处理的是 underactuated objects，也就是对象本身有独立动力学和非完整约束，不是被机器人末端完全控制的刚性物体。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#38/42）。
- HAIC 关注 Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。它处理的是 underactuated objects，也就是对象本身有独立动力学和非完整约束，不是被机器人末端完全控制的刚性物体。
- 这类任务很难。比如推车、搬运、球类、箱体互动、受遮挡物体，机器人不能只根据当前几何位置规划手臂动作。对象会受到力、摩擦、惯性、接触约束影响，反过来改变机器人状态。
- HAIC 的核心是 dynamics-aware world model，用来预测对象和机器人之间的交互后果，从而支持控制。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 38/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | 清华大学；香港科技大学（广州）；苏黎世联邦理工；小米机器人实验室 |
| 出处 | curated |
| 链接 | <https://haic-humanoid.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

HAIC 关注 Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。它处理的是 underactuated objects，也就是对象本身有独立动力学和非完整约束，不是被机器人末端完全控制的刚性物体。

### 2）策展导读要点

这类任务很难。比如推车、搬运、球类、箱体互动、受遮挡物体，机器人不能只根据当前几何位置规划手臂动作。对象会受到力、摩擦、惯性、接触约束影响，反过来改变机器人状态。

### 3）策展导读要点

HAIC 的核心是 dynamics-aware world model，用来预测对象和机器人之间的交互后果，从而支持控制。

### 4）策展导读要点

我把它放在 CHIP 和 GentleHumanoid 之后，是因为它把“接触”从机器人末端扩展到对象动力学。柔顺控制解决的是机器人怎么接触，HAIC 进一步问：接触之后对象会怎么动。

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md](../../sources/papers/humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md)

## 参考来源

- [humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md](../../sources/papers/humanoid_rl_stack_38_haic_humanoid_agile_object_interaction_control_v.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：HAIC](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/HAIC__Humanoid_Agile_Object_Interaction_Control_via_Dynamics-Aware_World_Model/HAIC__Humanoid_Agile_Object_Interaction_Control_via_Dynamics-Aware_World_Model.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
