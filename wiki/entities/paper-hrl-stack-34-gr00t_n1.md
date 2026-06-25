---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, nvidia]
status: complete
updated: 2026-06-25
venue: "project"
code: https://github.com/NVIDIA/Isaac-GR00T
summary: "GR00T N1 是这次新增论文里很关键的一篇。它的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../entities/gr00t-wholebodycontrol.md
sources:
  - ../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# GR00T N1

**GR00T N1** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 34/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。

> **深读页：** [gr00t-wholebodycontrol](../entities/gr00t-wholebodycontrol.md) — 方法机制与实验细节见链接页；本页保留 survey 坐标与交叉引用。

## 一句话定义

GR00T N1 是这次新增论文里很关键的一篇。它的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#34/42）。
- GR00T N1 是这次新增论文里很关键的一篇。它的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。
- 这听起来像“VLA 控制机器人”，但论文真正有价值的地方恰恰不是这句口号，而是它把机器人动作接口拆得很具体：状态历史、action chunk、embodiment tag、latent action、真实动作标签、合成轨迹和真机 post-training 都要一起进入系统。
- 最底层是大量人类视频和网络视频，中间是神经生成轨迹和仿真轨迹，最上层才是真机机器人数据。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 34/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | NVIDIA 等 |
| 出处 | project |
| 链接 | <https://github.com/NVIDIA/Isaac-GR00T> |

## 核心机制（归纳）

### 1）策展导读要点

GR00T N1 是这次新增论文里很关键的一篇。它的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。

### 2）策展导读要点

这听起来像“VLA 控制机器人”，但论文真正有价值的地方恰恰不是这句口号，而是它把机器人动作接口拆得很具体：状态历史、action chunk、embodiment tag、latent action、真实动作标签、合成轨迹和真机 post-training 都要一起进入系统。

### 3）策展导读要点

最底层是大量人类视频和网络视频，中间是神经生成轨迹和仿真轨迹，最上层才是真机机器人数据。

### 4）策展导读要点

这样做的原因很现实：真机数据最贵、最贴近身体，但覆盖度不足；人类视频和合成数据覆盖度高，但必须通过 latent action、inverse dynamics 或 post-training 才能落到具体机器人身上。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md](../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md)

## 参考来源

- [humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md](../../sources/papers/humanoid_rl_stack_34_gr00t_n1_an_open_foundation_model_for_generalist.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：GR00T N1 Humanoid Foundation Model](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/GR00T_N1_Humanoid_Foundation_Model/GR00T_N1_Humanoid_Foundation_Model.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
