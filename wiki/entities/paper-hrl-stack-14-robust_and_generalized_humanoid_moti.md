---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-05-26
venue: curated
summary: "RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 anne"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Robust and Generalized Humanoid Motion Tracking

**Robust and Generalized Humanoid Motion Tracking** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 14/42** 篇，归类为 **02 参考跟踪 · 通用控制**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **02 参考跟踪 · 通用控制** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 14/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 北京理工大学；人形机器人（上海）有限公司 |
| 出处 | curated |
| 链接 | <https://zeonsunlightyu.github.io/RGMT.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md](../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。


## 参考来源

- [humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md](../../sources/papers/humanoid_rl_stack_14_robust_and_generalized_humanoid_motion_tracking.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
