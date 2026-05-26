---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-05-26
summary: "这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# Ego-Vision World Model for Humanoid Contact Planning

**Ego-Vision World Model for Humanoid Contact Planning** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 33/42** 篇，归类为 **04 视觉闭环 · 任务接口 · 世界模型**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- 这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **04 视觉闭环 · 任务接口 · 世界模型** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 33/42 |
| 系统栈层 | 04 视觉闭环 · 任务接口 · 世界模型 |
| 机构 | 伯克利；密歇根大学安娜堡分校；香港中文大学 |
| 出处 | curated |
| 链接 | <https://ego-vcp.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
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
