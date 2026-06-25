---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, umich, cuhk, berkeley]
status: complete
updated: 2026-06-25
venue: curated
summary: "这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/ego-9-papers-technology-map.md
  - ../overview/ego-category-03-world-models.md
sources:
  - ../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_survey_06_ego_vision_world_model.md
---

# Ego-Vision World Model for Humanoid Contact Planning

**Ego-Vision World Model for Humanoid Contact Planning** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 33/42** 篇（**04 视觉闭环 · 任务接口 · 世界模型**），并列入 [Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 06/9** 篇（**03 世界模型**）。

## 一句话定义

这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **04 视觉闭环 · 任务接口 · 世界模型**（#33/42）。
- 这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。
- 它解决的问题是：传统 optimization-based contact planning 面对复杂接触时很难扩展，online RL 又样本效率低。论文用 demonstration-free offline dataset 训练 world model，在压缩 latent space 中预测任务结果，并结合 value function 做更密集、更鲁棒的 planning。
- 我把它放在世界模型主线里，是因为它说明 world model 不一定只是视觉模型，也可以是身体接触模型。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 33/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | 伯克利；密歇根大学安娜堡分校；香港中文大学 |
| 出处 | curated |
| 链接 | <https://ego-vcp.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。

### 2）策展导读要点

它解决的问题是：传统 optimization-based contact planning 面对复杂接触时很难扩展，online RL 又样本效率低。论文用 demonstration-free offline dataset 训练 world model，在压缩 latent space 中预测任务结果，并结合 value function 做更密集、更鲁棒的 planning。

### 3）策展导读要点

我把它放在世界模型主线里，是因为它说明 world model 不一定只是视觉模型，也可以是身体接触模型。

### 4）策展导读要点

未来人形机器人进入复杂环境后，很多任务都依赖主动接触：扶墙、撑桌、挡物体、钻过低矮空间、利用环境恢复平衡。世界模型如果能预测这些接触后果，就能为高层控制提供重要能力。

## 常见误区

1. VLA/世界模型条目解决 **接口与预测**，不自动替代已封装的底层 WBC 能力。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- Ego 专题：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)、[ego-category-03-world-models.md](../overview/ego-category-03-world-models.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md](../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md)

## 参考来源

- [humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md](../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
