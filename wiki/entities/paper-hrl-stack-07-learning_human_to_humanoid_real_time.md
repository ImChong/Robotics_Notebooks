---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, cmu]
status: complete
updated: 2026-06-25
venue: curated
summary: "H2O 的完整名字是 Human-to-Humanoid。它要解决的问题很直接：能不能让一个人通过自己的身体动作，实时驱动一个人形机器人做全身动作，而不是只控制手臂或轮式底盘。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_07_learning_human_to_humanoid_real_time_whole_body.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation

**Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 07/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

## 一句话定义

H2O 的完整名字是 Human-to-Humanoid。它要解决的问题很直接：能不能让一个人通过自己的身体动作，实时驱动一个人形机器人做全身动作，而不是只控制手臂或轮式底盘。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#07/42）。
- H2O 的完整名字是 Human-to-Humanoid。它要解决的问题很直接：能不能让一个人通过自己的身体动作，实时驱动一个人形机器人做全身动作，而不是只控制手臂或轮式底盘。
- 这篇论文放在 HumanX 和 HDMI 后面很合适。HumanX / HDMI 关心的是从人类视频里生产可训练的交互技能，H2O 则更进一步，把“人类动作到机器人身体”的转换做成实时遥操作系统。
- 它的关键不只是 pose retargeting，而是把数据、仿真和真机控制接成一条闭环。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 07/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | CMU |
| 出处 | curated |
| 链接 | <https://human2humanoid.com/> |

## 核心机制（归纳）

### 1）策展导读要点

H2O 的完整名字是 Human-to-Humanoid。它要解决的问题很直接：能不能让一个人通过自己的身体动作，实时驱动一个人形机器人做全身动作，而不是只控制手臂或轮式底盘。

### 2）策展导读要点

这篇论文放在 HumanX 和 HDMI 后面很合适。HumanX / HDMI 关心的是从人类视频里生产可训练的交互技能，H2O 则更进一步，把“人类动作到机器人身体”的转换做成实时遥操作系统。

### 3）策展导读要点

它的关键不只是 pose retargeting，而是把数据、仿真和真机控制接成一条闭环。

### 4）策展导读要点

论文先从 AMASS 这类人体动作数据出发，经过 SMPL 到 Unitree H1 的重定向，再用 privileged motion imitator 过滤掉机器人动力学上不可执行的动作，形成 feasible motion dataset。之后再训练只依赖真实部署时可获得状态的 imitator，让机器人在真实世界里跟随人体动作。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_07_learning_human_to_humanoid_real_time_whole_body.md](../../sources/papers/humanoid_rl_stack_07_learning_human_to_humanoid_real_time_whole_body.md)

## 参考来源

- [humanoid_rl_stack_07_learning_human_to_humanoid_real_time_whole_body.md](../../sources/papers/humanoid_rl_stack_07_learning_human_to_humanoid_real_time_whole_body.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
