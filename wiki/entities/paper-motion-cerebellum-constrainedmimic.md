---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
venue: curated
summary: "安全：给 motion tracking 加实时约束层。输入是 motion tracking 策略输出和实时安全约束；实现上在策略动作外叠加约束投影/安全过滤，让跟踪参考时不突破关节、接触或碰撞边界；目标是保留动作表现同时守住安全。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-04-wbt-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_35_constrainedmimic.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# ConstrainedMimic

**ConstrainedMimic** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 35/64** 篇，归类为 **D 全身跟踪基座**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 安全：给 motion tracking 加实时约束层。输入是 motion tracking 策略输出和实时安全约束；实现上在策略动作外叠加约束投影/安全过滤，让跟踪参考时不突破关节、接触或碰撞边界；目标是保留动作表现同时守住安全。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[全身跟踪基座](../overview/motion-cerebellum-category-04-wbt-base.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 35/64 |
| 分组 | D 全身跟踪基座 |
| 机构 | 斯坦福大学、英伟达研究院 |
| 论文/项目 | https://danielpmorton.github.io/ |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-04-wbt-base.md](../overview/motion-cerebellum-category-04-wbt-base.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息（索引级）** 表）。
- 如需与运动小脑同组篇目对照实验，请回到 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 对应分类 hub 的评测段落。

## 参考来源

- [motion_cerebellum_survey_35_constrainedmimic.md](../../sources/papers/motion_cerebellum_survey_35_constrainedmimic.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
