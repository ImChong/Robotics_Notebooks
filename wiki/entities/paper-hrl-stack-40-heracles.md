---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, x-humanoid]
status: complete
updated: 2026-06-25
venue: curated
summary: "Heracles 试图连接 precise tracking 和 generative synthesis。它的核心问题是：传统 tracking controller 在正常状态下很好，但当机器人受到强扰动、摔倒或远离参考状态时，继续追踪原始参考动作可能会更糟。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_40_heracles_bridging_precise_tracking_and_generativ.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Heracles

**Heracles** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 40/42** 篇，归类为 **05 接触 · 柔顺 · 安全恢复**。

> **深读页：** [paper-heracles-humanoid-diffusion](../entities/paper-heracles-humanoid-diffusion.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

Heracles 试图连接 precise tracking 和 generative synthesis。它的核心问题是：传统 tracking controller 在正常状态下很好，但当机器人受到强扰动、摔倒或远离参考状态时，继续追踪原始参考动作可能会更糟。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **05 接触 · 柔顺 · 安全恢复**（#40/42）。
- Heracles 试图连接 precise tracking 和 generative synthesis。它的核心问题是：传统 tracking controller 在正常状态下很好，但当机器人受到强扰动、摔倒或远离参考状态时，继续追踪原始参考动作可能会更糟。
- Heracles 在参考动作和低层物理跟踪器之间插入一个生成式中间件。正常时，它尽量保持参考动作；异常时，它根据当前状态生成新的未来关键姿态，让低层控制器执行更适合恢复的动作。
- 这篇论文最重要的思想是“异常状态下要改写参考”。一个人被推倒后，不会继续执行原来的走路动作，而会做出恢复动作。人形机器人也需要这种能力。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 40/42 |
| 系统栈层 | 05 接触 · 柔顺 · 安全恢复 |
| 机构 | X-Humanoid Heracles 项目组 / 北京人形机器人创新中心 |
| 出处 | curated |
| 链接 | <https://heracles-humanoid-control.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

Heracles 试图连接 precise tracking 和 generative synthesis。它的核心问题是：传统 tracking controller 在正常状态下很好，但当机器人受到强扰动、摔倒或远离参考状态时，继续追踪原始参考动作可能会更糟。

### 2）策展导读要点

Heracles 在参考动作和低层物理跟踪器之间插入一个生成式中间件。正常时，它尽量保持参考动作；异常时，它根据当前状态生成新的未来关键姿态，让低层控制器执行更适合恢复的动作。

### 3）策展导读要点

这篇论文最重要的思想是“异常状态下要改写参考”。一个人被推倒后，不会继续执行原来的走路动作，而会做出恢复动作。人形机器人也需要这种能力。

### 4）策展导读要点

我的判断**通用控制器不能只会正常动作，还必须知道什么时候放弃原参考、重新生成恢复动作。**

## 常见误区

1. 柔顺/恢复策略要在 **接触丰富** 与 **长期稳定** 间折中，不能只看单帧姿态。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_40_heracles_bridging_precise_tracking_and_generativ.md](../../sources/papers/humanoid_rl_stack_40_heracles_bridging_precise_tracking_and_generativ.md)

## 参考来源

- [humanoid_rl_stack_40_heracles_bridging_precise_tracking_and_generativ.md](../../sources/papers/humanoid_rl_stack_40_heracles_bridging_precise_tracking_and_generativ.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
