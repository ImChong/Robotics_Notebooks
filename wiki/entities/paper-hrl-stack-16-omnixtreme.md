---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-06-11
venue: curated
summary: "OmniXtreme 的关键词是 generality barrier in high-dynamic humanoid control。它指出，当动作库越来越多、动作越来越极端时，通用性和跟踪精度之间会出现冲突。一个策略想覆盖更多动作，可能会损失高动态技能的执行质量。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_16_omnixtreme_breaking_the_generality_barrier_in_hi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# OmniXtreme

**OmniXtreme** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 16/42** 篇，归类为 **02 参考跟踪 · 通用控制**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- OmniXtreme 的关键词是 generality barrier in high-dynamic humanoid control。它指出，当动作库越来越多、动作越来越极端时，通用性和跟踪精度之间会出现冲突。一个策略想覆盖更多动作，可能会损失高动态技能的执行质量。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **02 参考跟踪 · 通用控制** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 16/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | BIGAI；BIGAI & 宇树科技；上海交通大学；中科大；宇树科技；华中科技大学；北京理工大学 |
| 出处 | curated |
| 链接 | <https://extreme-humanoid.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_16_omnixtreme_breaking_the_generality_barrier_in_hi.md](../../sources/papers/humanoid_rl_stack_16_omnixtreme_breaking_the_generality_barrier_in_hi.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_16_omnixtreme_breaking_the_generality_barrier_in_hi.md](../../sources/papers/humanoid_rl_stack_16_omnixtreme_breaking_the_generality_barrier_in_hi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：OmniXtreme](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/OmniXtreme_Breaking_the_Generality_Barrier_in_High-Dynamic_Humanoid_Control/OmniXtreme_Breaking_the_Generality_Barrier_in_High-Dynamic_Humanoid_Control.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
