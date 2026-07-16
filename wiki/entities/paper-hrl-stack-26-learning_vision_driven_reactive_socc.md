---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, bytedance, cau, tsinghua]
status: complete
updated: 2026-07-16
venue: curated
summary: "这篇论文要让人形机器人学习视觉驱动的反应式足球技能。它不是做一个传统规则系统，而是把视觉感知、运动先验和动态控制结合起来，让机器人在真实 RoboCup 类场景中完成更连贯的踢球行为。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots

**Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 26/42** 篇，归类为 **03 感知式高动态运动**。

## 一句话定义

这篇论文要让人形机器人学习视觉驱动的反应式足球技能。它不是做一个传统规则系统，而是把视觉感知、运动先验和动态控制结合起来，让机器人在真实 RoboCup 类场景中完成更连贯的踢球行为。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **03 感知式高动态运动**（#26/42）。
- 这篇论文要让人形机器人学习视觉驱动的反应式足球技能。它不是做一个传统规则系统，而是把视觉感知、运动先验和动态控制结合起来，让机器人在真实 RoboCup 类场景中完成更连贯的踢球行为。
- 论文提出 virtual perception system，模拟真实视觉误差，并用 encoder-decoder 从不完美观测中恢复 ball position 等 privileged-like state。这是为了解决真实视觉检测和控制之间的错位。
- 最有意思的是主动感知。机器人不是被动接收球的位置，而会调整躯干、头部和身体，让球保持在更好的视野里。也就是说，“看球”本身成为动作策略的一部分。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 26/42 |
| 系统栈层 | 03 感知式高动态运动 |
| 机构 | 清华大学；字节跳动 Seed；中国农业大学 |
| 出处 | curated |
| 链接 | <https://humanoid-kick.github.io> |

## 核心机制（归纳）

### 1）策展导读要点

这篇论文要让人形机器人学习视觉驱动的反应式足球技能。它不是做一个传统规则系统，而是把视觉感知、运动先验和动态控制结合起来，让机器人在真实 RoboCup 类场景中完成更连贯的踢球行为。

### 2）策展导读要点

论文提出 virtual perception system，模拟真实视觉误差，并用 encoder-decoder 从不完美观测中恢复 ball position 等 privileged-like state。这是为了解决真实视觉检测和控制之间的错位。

### 3）策展导读要点

最有意思的是主动感知。机器人不是被动接收球的位置，而会调整躯干、头部和身体，让球保持在更好的视野里。也就是说，“看球”本身成为动作策略的一部分。

## 常见误区

1. 感知 locomotion 的难点在 **闭环时延与几何误差**，不是单纯「加相机输入」。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md](../../sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md)

## 参考来源

- [humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md](../../sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
