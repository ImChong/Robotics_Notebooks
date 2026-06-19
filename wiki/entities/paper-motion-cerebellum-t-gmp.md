---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
venue: curated
summary: "底座：用地形条件运动先验改善自然步态。输入是地形条件和运动先验；实现上训练地形条件生成式运动先验，再用它约束强化学习策略生成自然多地形步态；重点是让步态风格和地形适配同时成立。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-01-locomotion-base.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_02_t_gmp.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# T-GMP

**T-GMP** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 02/64** 篇，归类为 **A 走路底座**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 底座：用地形条件运动先验改善自然步态。输入是地形条件和运动先验；实现上训练地形条件生成式运动先验，再用它约束强化学习策略生成自然多地形步态；重点是让步态风格和地形适配同时成立。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[走路底座](../overview/motion-cerebellum-category-01-locomotion-base.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 02/64 |
| 分组 | A 走路底座 |
| 机构 | 哈尔滨工业大学、乐聚机器人 |
| 论文/项目 | https://t-gmp.github.io |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-01-locomotion-base.md](../overview/motion-cerebellum-category-01-locomotion-base.md)

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **原文 PDF / 项目页** 为准（链接见 [参考来源](#参考来源) 与上文 **核心信息（索引级）** 表）。
- 如需与运动小脑同组篇目对照实验，请回到 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 对应分类 hub 的评测段落。

## 参考来源

- [motion_cerebellum_survey_02_t_gmp.md](../../sources/papers/motion_cerebellum_survey_02_t_gmp.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
