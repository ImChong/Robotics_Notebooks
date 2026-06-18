---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
summary: "接口：EE-root 命令连接高层和全身控制。输入是根部运动目标、末端执行器目标和柔顺控制参数；实现上把根部控制与柔顺末端执行器解耦，再通过层级接口协调移动和操作；重点是降低手、脚、腰之间的强耦合，让上层更容易调用。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-07-loco-manip-interface.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_47_ceer.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# CEER

**CEER** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 47/64** 篇，归类为 **G Loco-Manip 接口**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 接口：EE-root 命令连接高层和全身控制。输入是根部运动目标、末端执行器目标和柔顺控制参数；实现上把根部控制与柔顺末端执行器解耦，再通过层级接口协调移动和操作；重点是降低手、脚、腰之间的强耦合，让上层更容易调用。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[Loco-Manip 接口](../overview/motion-cerebellum-category-07-loco-manip-interface.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 47/64 |
| 分组 | G Loco-Manip 接口 |
| 机构 | 机构待核对 |
| 论文/项目 | https://robotproject8.github.io/ceer\_page/ |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-07-loco-manip-interface.md](../overview/motion-cerebellum-category-07-loco-manip-interface.md)

## 参考来源

- [motion_cerebellum_survey_47_ceer.md](../../sources/papers/motion_cerebellum_survey_47_ceer.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
