---
type: overview
tags: [egocentric, ego-survey, category-hub, world-models, humanoid, contact]
status: complete
updated: 2026-06-01
summary: "Ego 9 篇专题 · 03 世界模型（2 篇）— 长时程须区分世界变化与自我运动；Ego-Vision WM 服务接触规划，WEM 解耦 world/ego 视频演化。"
related:
  - ./ego-9-papers-technology-map.md
  - ./ego-category-02-human-to-robot.md
  - ./ego-category-04-ego-exo-fusion.md
  - ./robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md
  - ../entities/paper-wem-world-ego-modeling.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md
  - ../../sources/papers/ego_9_papers_catalog.md
  - ../../sources/papers/ego_survey_06_ego_vision_world_model.md
  - ../../sources/papers/ego_survey_07_wem.md
---

# Ego 分类 03：世界模型

> **图谱分类节点**：**03 世界模型**；总地图见 [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)。

## 核心问题

**为什么世界模型也开始区分 ego 和 world？** 相机运动、身体运动与环境/物体变化混在同一预测流里，长时程 rollout 易漂移。

## 本组论文（2 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 06 | Ego-Vision World Model | [paper-hrl-stack-33](../entities/paper-hrl-stack-33-ego_vision_world_model_for_humanoid.md) | [source](../../sources/papers/ego_survey_06_ego_vision_world_model.md) |
| 07 | WEM | [paper-wem](../entities/paper-wem-world-ego-modeling.md) | [source](../../sources/papers/ego_survey_07_wem.md) |

## 关联页面

- [机器人世界模型训练闭环](./robot-world-models-training-loop-taxonomy.md)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 参考来源

- [wechat_embodied_ai_lab_ego_9_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md)
- [ego_9_papers_catalog.md](../../sources/papers/ego_9_papers_catalog.md)
