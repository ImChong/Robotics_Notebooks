---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-05-26
summary: "OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# OmniTrack

**OmniTrack** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 12/42** 篇，归类为 **02 参考跟踪 · 通用控制**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 为什么重要

- OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **02 参考跟踪 · 通用控制** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 12/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 机构 | 华中科技大学；BIGAI；上海交通大学 |
| 出处 | curated |
| 链接 | <https://omnitrack-humanoid.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md](../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md)

## 参考来源

- [humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md](../../sources/papers/humanoid_rl_stack_12_omnitrack_general_motion_tracking_via_physics_co.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
