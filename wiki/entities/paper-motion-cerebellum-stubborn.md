---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
summary: "恢复：把跟踪和跌倒恢复放进统一 RL。输入是参考动作、跌倒状态和恢复奖励；实现上用统一强化学习同时训练跟踪与起身恢复，而不是把恢复做成独立状态机；目标是动作被打断后能继续回到可控状态。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_34_stubborn.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# Stubborn

**Stubborn** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 34/64** 篇，归类为 **D 全身跟踪基座**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 恢复：把跟踪和跌倒恢复放进统一 RL。输入是参考动作、跌倒状态和恢复奖励；实现上用统一强化学习同时训练跟踪与起身恢复，而不是把恢复做成独立状态机；目标是动作被打断后能继续回到可控状态。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[全身跟踪基座](../overview/motion-cerebellum-category-04-wbt-base.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 34/64 |
| 分组 | D 全身跟踪基座 |
| 机构 | 南方科技大学 |
| 论文/项目 | https://aislab-sustech.github.io/Stubborn/ |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-04-wbt-base.md](../overview/motion-cerebellum-category-04-wbt-base.md)

## 参考来源

- [motion_cerebellum_survey_34_stubborn.md](../../sources/papers/motion_cerebellum_survey_34_stubborn.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
