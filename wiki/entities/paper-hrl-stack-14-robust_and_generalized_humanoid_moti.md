---

type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, openloong, bit]
status: complete
updated: 2026-07-16
venue: curated
summary: "RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 anne"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Robust and Generalized Humanoid Motion Tracking

**Robust and Generalized Humanoid Motion Tracking** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 14/42** 篇，归类为 **02 参考跟踪 · 通用控制**。

## 一句话定义

RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **02 参考跟踪 · 通用控制**（#14/42）。
- RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。
- 参考动作可能来自不同数据源，可能有局部错误，也可能和当前动力学状态冲突。控制器如果一味追踪，就容易失稳。RGMT 让策略根据当前身体状态动态判断参考片段的重要性，从而降低不一致参考对控制的伤害。
- 我的判断**通用运动控制的下一步不是更强 tracking loss，而是更聪明地处理参考质量。**

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 14/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 北京理工大学；人形机器人（上海）有限公司 |
| 出处 | curated |
| 链接 | <https://zeonsunlightyu.github.io/RGMT.github.io/> |

## 核心机制（归纳）

### 1）策展导读要点

RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。

### 2）策展导读要点

参考动作可能来自不同数据源，可能有局部错误，也可能和当前动力学状态冲突。控制器如果一味追踪，就容易失稳。RGMT 让策略根据当前身体状态动态判断参考片段的重要性，从而降低不一致参考对控制的伤害。

### 3）策展导读要点

我的判断**通用运动控制的下一步不是更强 tracking loss，而是更聪明地处理参考质量。**

## 常见误区

1. Motion tracking 论文的泛化常指 **参考分布内**；换数据源或接触条件仍可能崩塌。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md](../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md)

## 参考来源

- [humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md](../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Robust and Generalized Humanoid Motion Tracking](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Robust_and_Generalized_Humanoid_Motion_Tracking/Robust_and_Generalized_Humanoid_Motion_Tracking.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
