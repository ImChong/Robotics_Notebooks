---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, ut-austin, hkust, nvidia, uw, stanford, kaist, berkeley]
status: complete
updated: 2026-07-16
venue: curated
summary: "DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_35_dreamdojo_a_generalist_robot_world_model_from_la.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# DreamDojo

**DreamDojo** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 35/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

## 一句话定义

DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#35/42）。
- DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。
- 这里最关键的不是“会生成视频”，而是它试图让世界模型具备机器人策略可用的物理和动作可控性。
- 论文用大规模第一视角人类视频做预训练，数据规模达到 44K 小时；然后用 continuous latent action 解决人类视频没有机器人动作标签的问题；最后再用少量目标机器人数据 post-train，让模型对具体机器人 embodiment 和 action space 变得可控。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 35/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | NVIDIA；香港科技大学；伯克利；华盛顿大学；斯坦福大学；KAIST；德州大学奥斯汀分校等 |
| 出处 | curated |
| 链接 | <https://dreamdojo-world.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。

### 2）策展导读要点

这里最关键的不是“会生成视频”，而是它试图让世界模型具备机器人策略可用的物理和动作可控性。

### 3）策展导读要点

论文用大规模第一视角人类视频做预训练，数据规模达到 44K 小时；然后用 continuous latent action 解决人类视频没有机器人动作标签的问题；最后再用少量目标机器人数据 post-train，让模型对具体机器人 embodiment 和 action space 变得可控。

### 4）策展导读要点

DreamDojo 的下游实验也很有意思。它不只是拿世界模型做长视频生成，而是拿来做 live teleoperation、policy evaluation 和 model-based planning。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_35_dreamdojo_a_generalist_robot_world_model_from_la.md](../../sources/papers/humanoid_rl_stack_35_dreamdojo_a_generalist_robot_world_model_from_la.md)

## 参考来源

- [humanoid_rl_stack_35_dreamdojo_a_generalist_robot_world_model_from_la.md](../../sources/papers/humanoid_rl_stack_35_dreamdojo_a_generalist_robot_world_model_from_la.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：DreamDojo](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DreamDojo_A_Generalist_Robot_World_Model_from_Large-Scale_Human_Videos/DreamDojo_A_Generalist_Robot_World_Model_from_Large-Scale_Human_Videos.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
