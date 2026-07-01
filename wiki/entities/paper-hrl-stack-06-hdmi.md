---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, cmu]
status: complete
updated: 2026-07-01
venue: curated
summary: "HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_06_hdmi_learning_interactive_humanoid_whole_body_co.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# HDMI

**HDMI** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 06/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。

## 一句话定义

HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **01 数据 · 重定向 · 遥操作**（#06/42）。
- HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。
- 这篇论文要解决的是：如果视频里有人推箱子、搬箱子、开门、滚球、放倒木板，人形机器人能不能把这些人-物交互变成自己可以执行的全身技能？
- HDMI 的管线分成三步：先从单目 RGB 视频里估计人体和物体轨迹，并重定向成人形机器人参考数据；再用强化学习训练一个 robot-object co-tracking policy，让机器人同时跟踪自己的身体状态和物体状态；最后把策略零样本部署到真实 Unitree G1 上。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 06/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | CMU |
| 出处 | curated |
| 链接 | <https://hdmi-humanoid.github.io> |

## 核心机制（归纳）

### 1）策展导读要点

HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。

### 2）策展导读要点

这篇论文要解决的是：如果视频里有人推箱子、搬箱子、开门、滚球、放倒木板，人形机器人能不能把这些人-物交互变成自己可以执行的全身技能？

### 3）策展导读要点

HDMI 的管线分成三步：先从单目 RGB 视频里估计人体和物体轨迹，并重定向成人形机器人参考数据；再用强化学习训练一个 robot-object co-tracking policy，让机器人同时跟踪自己的身体状态和物体状态；最后把策略零样本部署到真实 Unitree G1 上。

### 4）策展导读要点

这里最关键的变化是，控制器不再只追踪“机器人像不像人”，还要追踪“物体有没有按预期被改变”。这就把模仿学习从 body motion tracking 推到了 object interaction tracking。

## 常见误区

1. 重定向/遥操作不是「训练前脚本」——参考质量上限往往 **早于** RL 策略决定。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_06_hdmi_learning_interactive_humanoid_whole_body_co.md](../../sources/papers/humanoid_rl_stack_06_hdmi_learning_interactive_humanoid_whole_body_co.md)

## 参考来源

- [humanoid_rl_stack_06_hdmi_learning_interactive_humanoid_whole_body_co.md](../../sources/papers/humanoid_rl_stack_06_hdmi_learning_interactive_humanoid_whole_body_co.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
