---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, cross-embodiment-teleop]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · F 跨本体与遥操作（5 篇）— 跨本体等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-any2any-cross-embodiment-wbt.md
  - ../entities/paper-twist.md
  - ../entities/paper-twist2.md
  - ../entities/paper-loco-manip-08-x-op.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 F：跨本体与遥操作

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **F 跨本体与遥操作** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

跨本体：把预训练 whole-body tracker 迁到新身体。输入是源机器人或人体动作和目标机器人形态；实现上学习跨本体运动迁移，把动作意图、接触和根部运动转成目标机器人可跟踪轨迹；重点是减少每换一台机器人就重新整理数据的成本。

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 42 | Any2Any | [paper-any2any-cross-embodiment-wbt.md](../entities/paper-any2any-cross-embodiment-wbt.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 43 | TWIST | [paper-twist.md](../entities/paper-twist.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 44 | TWIST2 | [paper-twist2.md](../entities/paper-twist2.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 45 | X-OP | [paper-loco-manip-08-x-op.md](../entities/paper-loco-manip-08-x-op.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 46 | CLOT | [paper-amp-survey-16-clot.md](../entities/paper-amp-survey-16-clot.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
