---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-06-01
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

**Ego-Vision World Model for Humanoid Contact Planning** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 33/42** 篇（**04 视觉闭环 · 任务接口 · 世界模型**），并列入 [Ego 9 篇专题](https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA) **第 06/9** 篇（**03 世界模型**）。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

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
- Ego 专题：[ego-9-papers-technology-map.md](../overview/ego-9-papers-technology-map.md)、[ego-category-03-world-models.md](../overview/ego-category-03-world-models.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md](../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md](../../sources/papers/humanoid_rl_stack_33_ego_vision_world_model_for_humanoid_contact_plan.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
