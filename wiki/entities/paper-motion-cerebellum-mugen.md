---
type: entity
tags: [paper, motion-cerebellum-survey, humanoid, motion-control]
status: complete
updated: 2026-06-18
summary: "可提示小脑：多技能生成式运动控制。输入是多技能运动数据和任务条件；实现上用 VQ-VAE 离散化运动技能，再结合模型式 RL 和师生蒸馏训练统一控制器；目标是让多个 locomotion 技能共享一个生成式动作空间。"
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-05-promptable-control.md
sources:
  - ../../sources/papers/motion_cerebellum_survey_39_mugen.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# MuGen

**MuGen** 收录于 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) **第 39/64** 篇，归类为 **E 可提示控制**。本页为知识库 **策展摘要**；方法细节以论文 PDF 与项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |

## 为什么重要

- 可提示小脑：多技能生成式运动控制。输入是多技能运动数据和任务条件；实现上用 VQ-VAE 离散化运动技能，再结合模型式 RL 和师生蒸馏训练统一控制器；目标是让多个 locomotion 技能共享一个生成式动作空间。
- 在 [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中属于 **[可提示控制](../overview/motion-cerebellum-category-05-promptable-control.md)**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 39/64 |
| 分组 | E 可提示控制 |
| 机构 | 北京大学 |
| 论文/项目 | https://arxiv.org/abs/2605.24592v1 |

## 与其他页面的关系

- 技术地图：[humanoid-motion-cerebellum-technology-map.md](../overview/humanoid-motion-cerebellum-technology-map.md)
- 分类 hub：[motion-cerebellum-category-05-promptable-control.md](../overview/motion-cerebellum-category-05-promptable-control.md)

## 参考来源

- [motion_cerebellum_survey_39_mugen.md](../../sources/papers/motion_cerebellum_survey_39_mugen.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)
- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 推荐继续阅读

- [运动小脑技术地图](../overview/humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
