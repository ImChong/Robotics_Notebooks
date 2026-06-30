---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, real-tasks]
status: complete
updated: 2026-06-30
summary: "运动小脑 64 篇长文 · H 真实任务（8 篇）— 任务等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/paper-doorman-opening-sim2real-door.md
  - ../entities/paper-motion-cerebellum-hoist.md
  - ../entities/paper-splitadapter-load-aware-loco-manipulation.md
  - ../entities/paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 H：真实任务

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **H 真实任务** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

任务：开门把视觉、接触、移动和平衡全照出来。输入是 RGB 图像和开门任务状态；实现上在仿真中训练 pixel-to-action 策略，再通过视觉随机化和真实部署适配完成开门；重点是从像素直接迁移到人形真机操作。

## 本组论文（8 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 52 | DoorMan | [paper-doorman-opening-sim2real-door.md](../entities/paper-doorman-opening-sim2real-door.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 53 | HOIST | [paper-motion-cerebellum-hoist.md](../entities/paper-motion-cerebellum-hoist.md) | [catalog](../../sources/papers/motion_cerebellum_survey_53_hoist.md) |
| 54 | SplitAdapter | [paper-splitadapter-load-aware-loco-manipulation.md](../entities/paper-splitadapter-load-aware-loco-manipulation.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 55 | HALO | [paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md](../entities/paper-hrl-stack-39-closing_sim_to_real_gap_for_heavy_lo.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 56 | HumanoidMimicGen | [paper-motion-cerebellum-humanoidmimicgen.md](../entities/paper-motion-cerebellum-humanoidmimicgen.md) | [catalog](../../sources/papers/motion_cerebellum_survey_56_humanoidmimicgen.md) |
| 57 | GRAIL | [paper-grail.md](../entities/paper-grail.md) | [catalog](../../sources/papers/motion_cerebellum_survey_57_grail.md) |
| 58 | OASIS | [paper-loco-manip-04-oasis.md](../entities/paper-loco-manip-04-oasis.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 59 | LadderMan | [paper-ladderman-humanoid-perceptive-ladder-climbing.md](../entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
