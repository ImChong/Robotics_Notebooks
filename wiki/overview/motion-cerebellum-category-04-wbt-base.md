---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, wbt-base]
status: complete
updated: 2026-07-16
summary: "运动小脑 64 篇长文 · D 全身跟踪基座（13 篇）— 跟踪策略等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-hrl-stack-12-omnitrack.md
  - ../methods/beyondmimic.md
  - ../methods/sonic-motion-tracking.md
  - ../entities/holomotion.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 D：全身跟踪基座

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **D 全身跟踪基座** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

跟踪策略：物理一致参考让 tracking 更稳。输入是参考运动和物理一致性约束；实现上先修正参考轨迹，让接触、速度和根部运动更符合动力学，再训练跟踪策略；重点是让策略学到的不是几何姿态，而是物理可执行动作。

## 本组论文（13 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 25 | OmniTrack | [paper-hrl-stack-12-omnitrack.md](../entities/paper-hrl-stack-12-omnitrack.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 26 | BeyondMimic | [beyondmimic.md](../methods/beyondmimic.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 27 | SONIC | [sonic-motion-tracking.md](../methods/sonic-motion-tracking.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 28 | HoloMotion-1 | [holomotion.md](../entities/holomotion.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 29 | HumanoidGPT | [paper-humanoid-gpt.md](../entities/paper-humanoid-gpt.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 30 | LIMMT | [limmt-gqs-motion-curation.md](../methods/limmt-gqs-motion-curation.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 31 | M3imic | [paper-loco-manip-06-m3imic.md](../entities/paper-loco-manip-06-m3imic.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 32 | RGMT | [paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md](../entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 33 | Any2Track | [any2track.md](../methods/any2track.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 34 | Stubborn | [paper-motion-cerebellum-stubborn.md](../entities/paper-motion-cerebellum-stubborn.md) | [catalog](../../sources/papers/motion_cerebellum_survey_34_stubborn.md) |
| 35 | ConstrainedMimic | [paper-motion-cerebellum-constrainedmimic.md](../entities/paper-motion-cerebellum-constrainedmimic.md) | [catalog](../../sources/papers/motion_cerebellum_survey_35_constrainedmimic.md) |
| 36 | SafeWBC | [paper-motion-cerebellum-safewbc.md](../entities/paper-motion-cerebellum-safewbc.md) | [catalog](../../sources/papers/motion_cerebellum_survey_36_safewbc.md) |
| 37 | SafeFall | [paper-hrl-stack-41-safefall.md](../entities/paper-hrl-stack-41-safefall.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
