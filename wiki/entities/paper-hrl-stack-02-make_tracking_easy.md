---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack]
status: complete
updated: 2026-06-18
venue: curated
summary: "NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-03-data-pipeline.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../methods/neural-motion-retargeting-nmr.md
sources:
  - ../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
---

# Make Tracking Easy

**Make Tracking Easy** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 02/42** 篇，归类为 **01 数据 · 重定向 · 遥操作**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |

## 为什么重要

- 在 [运动小脑 64 篇技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中归类为 **C 数据入口**（19/64）：重定向：神经重定向与物理修正数据。
- NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。
- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 的八层框架中，属于 **01 数据 · 重定向 · 遥操作** 簇。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 02/42 |
| 系统栈层 | 01 数据 · 重定向 · 遥操作 |
| 机构 | 南京大学；华为 |
| 出处 | curated |
| 链接 | <https://nju3dv-humanoidgroup.github.io/nmr.github.io/> |

## 与其他页面的关系

- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP 姊妹篇：[humanoid-amp-motion-prior-survey.md](../overview/humanoid-amp-motion-prior-survey.md)
- 原始 source：[humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md](../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息** 表）。
- 若需与姊妹篇对照，请回到对应 **技术地图 / 42 篇栈 / AMP 专题** 总览中的实验段落。

## 参考来源

- [humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md](../../sources/papers/humanoid_rl_stack_02_make_tracking_easy_neural_motion_retargeting_for.md) — 42 篇栈策展摘录
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) — 微信公众号编译导读
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)

## 推荐继续阅读

- [机器人论文阅读笔记：Make Tracking Easy](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/02_Motion_Retargeting/Make_Tracking_Easy__Neural_Motion_Retargeting_for_Humanoid_Whole-body_Control/Make_Tracking_Easy__Neural_Motion_Retargeting_for_Humanoid_Whole-body_Control.html)
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [19 篇 AMP 运动先验姊妹篇](https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w)
