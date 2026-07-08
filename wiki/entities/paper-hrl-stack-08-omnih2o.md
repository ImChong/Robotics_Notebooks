---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, sjtu, cmu]
status: complete
updated: 2026-07-08
venue: curated
summary: "如果说 H2O 证明了实时 whole-body teleoperation 这条路能走，OmniH2O 就是在问：这条路能不能变成一个更通用的身体接口？"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_08_omnih2o_universal_and_dexterous_human_to_humanoi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# OmniH2O

**OmniH2O** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 08/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

## 一句话定义

如果说 H2O 证明了实时 whole-body teleoperation 这条路能走，OmniH2O 就是在问：这条路能不能变成一个更通用的身体接口？

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#08/42）。
- 如果说 H2O 证明了实时 whole-body teleoperation 这条路能走，OmniH2O 就是在问：这条路能不能变成一个更通用的身体接口？
- OmniH2O 的目标比 H2O 更大。它不只想让机器人跟随人的身体动作，还要支持灵巧手、移动操作、户外行走，以及多种输入来源：VR、RGB 摄像头、语言指令，以及 GPT-4o / diffusion policy 这类自主策略。论文里一个核心说法，是把 kinematic pose 当成 universal control interface。
- 因为上层系统最终很难直接输出每个关节的力矩。无论输入来自人、视频、运动生成模型还是学习策略，它都需要一个中间身体接口，把“想做什么”转换成机器人可以稳定执行的全身姿态和运动目标。等这个接口足够稳，语言模型和 VLA 才有可能可靠地接上来。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 08/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | CMU；上海交通大学 |
| 出处 | curated |
| 链接 | <https://omni.human2humanoid.com/> |

## 核心机制（归纳）

### 1）策展导读要点

如果说 H2O 证明了实时 whole-body teleoperation 这条路能走，OmniH2O 就是在问：这条路能不能变成一个更通用的身体接口？

### 2）策展导读要点

OmniH2O 的目标比 H2O 更大。它不只想让机器人跟随人的身体动作，还要支持灵巧手、移动操作、户外行走，以及多种输入来源：VR、RGB 摄像头、语言指令，以及 GPT-4o / diffusion policy 这类自主策略。论文里一个核心说法，是把 kinematic pose 当成 universal control interface。

### 3）策展导读要点

因为上层系统最终很难直接输出每个关节的力矩。无论输入来自人、视频、运动生成模型还是学习策略，它都需要一个中间身体接口，把“想做什么”转换成机器人可以稳定执行的全身姿态和运动目标。等这个接口足够稳，语言模型和 VLA 才有可能可靠地接上来。

### 4）策展导读要点

OmniH2O 试图把这个接口做宽：上层可以给稀疏输入，底层控制器负责补齐全身协调、接触稳定和灵巧手动作。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_08_omnih2o_universal_and_dexterous_human_to_humanoi.md](../../sources/papers/humanoid_rl_stack_08_omnih2o_universal_and_dexterous_human_to_humanoi.md)

## 参考来源

- [humanoid_rl_stack_08_omnih2o_universal_and_dexterous_human_to_humanoi.md](../../sources/papers/humanoid_rl_stack_08_omnih2o_universal_and_dexterous_human_to_humanoi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：OmniH2O Universal Whole-Body Teleoperation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/OmniH2O_Universal_Whole-Body_Teleoperation/OmniH2O_Universal_Whole-Body_Teleoperation.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
