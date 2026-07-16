---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, motion-imitation]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · B 动作模仿源流（5 篇）— 源流等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../methods/deepmimic.md
  - ../entities/paper-amp-survey-01-amp.md
  - ../methods/smp.md
  - ../entities/phc.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 B：动作模仿源流

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **B 动作模仿源流** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

源流：参考动作 + 强化学习。输入是参考动作片段和物理角色状态；实现上用强化学习最大化姿态、速度、末端、根部等跟踪奖励，并通过 early termination 保持物理合理；它奠定了物理角色模仿学习的基本范式。

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 11 | DeepMimic | [deepmimic.md](../methods/deepmimic.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 12 | AMP | [paper-amp-survey-01-amp.md](../entities/paper-amp-survey-01-amp.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 13 | SMP | [smp.md](../methods/smp.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 14 | PHC | [phc.md](../entities/phc.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 15 | MaskedMimic | [paper-bfm-17-maskedmimic.md](../entities/paper-bfm-17-maskedmimic.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
