---
type: overview
tags: [motion-cerebellum, humanoid, category-hub, survey, data-pipeline]
status: complete
updated: 2026-06-18
summary: "运动小脑 64 篇长文 · C 数据入口（9 篇）— 数据入口等站位。"
related:
  - ./humanoid-motion-cerebellum-technology-map.md
  - ../entities/gvhmr.md
  - ../entities/paper-motion-cerebellum-tram.md
  - ../entities/paper-hrl-stack-01-retargeting_matters.md
  - ../entities/paper-hrl-stack-02-make_tracking_easy.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
---

# 运动小脑分类 C：数据入口

> **图谱分类节点**：对应 [具身智能研究室 · 运动小脑 64 篇长文](https://mp.weixin.qq.com/s/Kx9myecE1Z0eGqOapoqQnA) 的 **C 数据入口** 分组；总地图见 [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| RL | Reinforcement Learning | 通过与环境交互学习策略的范式 |
| VLA | Vision-Language-Action | 上层策略调用身体 API 的典型形态 |

## 核心问题

数据入口：视频动作恢复到重力对齐世界坐标。输入是视频中的人体运动；实现上在重力对齐坐标系里恢复全局人体轨迹和姿态，减少相机视角带来的漂移；它给机器人重定向提供更稳定的世界系人体动作。

## 本组论文（9 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 16 | GVHMR | [gvhmr.md](../entities/gvhmr.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 17 | TRAM | [paper-motion-cerebellum-tram.md](../entities/paper-motion-cerebellum-tram.md) | [catalog](../../sources/papers/motion_cerebellum_survey_17_tram.md) |
| 18 | GMR | [paper-hrl-stack-01-retargeting_matters.md](../entities/paper-hrl-stack-01-retargeting_matters.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 19 | NMR | [paper-hrl-stack-02-make_tracking_easy.md](../entities/paper-hrl-stack-02-make_tracking_easy.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 20 | OmniRetarget | [paper-hrl-stack-03-omniretarget.md](../entities/paper-hrl-stack-03-omniretarget.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 21 | HumanX | [paper-hrl-stack-05-humanx.md](../entities/paper-hrl-stack-05-humanx.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 22 | HDMI | [paper-hrl-stack-06-hdmi.md](../entities/paper-hrl-stack-06-hdmi.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 23 | SUGAR | [paper-notebook-sugar-a-scalable-human-video-driven-generalizabl.md](../entities/paper-notebook-sugar-a-scalable-human-video-driven-generalizabl.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |
| 24 | GenMimic | [paper-hrl-stack-04-from_generated_human_videos_to_physi.md](../entities/paper-hrl-stack-04-from_generated_human_videos_to_physi.md) | [catalog](../../sources/papers/motion_cerebellum_64_catalog.md) |

## 关联页面

- [运动小脑技术地图](./humanoid-motion-cerebellum-technology-map.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
- [motion_cerebellum_64_catalog.md](../../sources/papers/motion_cerebellum_64_catalog.md)

## 推荐继续阅读

- [Loco-Manip 8 篇技术地图](./loco-manip-8-papers-technology-map.md)
