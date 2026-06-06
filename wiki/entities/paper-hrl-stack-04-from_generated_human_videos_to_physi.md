---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-05-26
venue: curated
summary: "这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
sources:
  - ../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
---

# From Generated Human Videos to Physically Plausible Robot Trajectories

**From Generated Human Videos to Physically Plausible Robot Trajectories** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 04/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **01 数据 · 重定向 · 遥操作** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 04/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 伯克利；纽约大学；约翰内斯开普勒大学 |
| 出处 | curated |
| 链接 | <https://genmimic.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md](../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md](../../sources/papers/humanoid_rl_stack_04_from_generated_human_videos_to_physically_plausi.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
