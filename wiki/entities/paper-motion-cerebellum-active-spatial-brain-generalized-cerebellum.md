---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
summary: "接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-07-loco-manip-interface.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 主动空间大脑与泛化动作小脑

**主动空间大脑与泛化动作小脑** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 51/64** 篇，归类为 **G Loco-Manip 接口**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 接口：上层规划与泛化动作小脑分工。输入是上层任务命令、物体/负载状态和机器人本体状态；实现上把移动、操作、平衡和负载扰动拆成可控接口，再通过蒸馏、MPC 引导或强化学习训练全身策略；重点是让 VLA/planner 可以稳定调用身体。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[Loco-Manip 接口](../overview/motion-cerebellum-category-07-loco-manip-interface.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 51/64 |
| 分组 | G Loco-Manip 接口 |
| 机构 | 中山大学计算机科学与工程学院 |
| 论文/项目 | https://leungchaos.github.io/Humanoid-Whole-Body-Manipulation-via-Active- |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-07-loco-manip-interface.md](../overview/motion-cerebellum-category-07-loco-manip-interface.md)

## 参考来源

- [motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md](../../sources/papers/motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
