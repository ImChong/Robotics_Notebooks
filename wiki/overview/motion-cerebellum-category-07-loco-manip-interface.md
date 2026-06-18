---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, loco-manip-interface]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · G Loco-Manip 接口（5 篇）— 接口等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-motion-cerebellum-ceer.md
  - ../entities/paper-motion-cerebellum-handoff.md
  - ../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md
  - ../entities/paper-loco-manip-05-vaic.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 G：Loco-Manip 接口

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **G Loco-Manip 接口** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

接口：EE-root 命令连接高层和全身控制。输入是根部运动目标、末端执行器目标和柔顺控制参数；实现上把根部控制与柔顺末端执行器解耦，再通过层级接口协调移动和操作；重点是降低手、脚、腰之间的强耦合，让上层更容易调用。

## 本组论文（5 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 47 | CEER | [paper-motion-cerebellum-ceer.md](../entities/paper-motion-cerebellum-ceer.md) | [catalog](../../sources/papers/motion_cerebellum_survey_47_ceer.md) |
| 48 | HANDOFF | [paper-motion-cerebellum-handoff.md](../entities/paper-motion-cerebellum-handoff.md) | [catalog](../../sources/papers/motion_cerebellum_survey_48_handoff.md) |
| 49 | MPC-RL | [paper-mpc-rl-humanoid-locomotion-manipulation.md](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 50 | VAIC | [paper-loco-manip-05-vaic.md](../entities/paper-loco-manip-05-vaic.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 51 | 主动空间大脑与泛化动作小脑 | [paper-motion-cerebellum-active-spatial-brain-generalized-cerebellum.md](../entities/paper-motion-cerebellum-active-spatial-brain-generalized-cerebellum.md) | [catalog](../../sources/papers/motion_cerebellum_survey_51_active_spatial_brain_generalized_cerebellum.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
