---
type: overview
tags: [world-models, world-action-models, category-hub, survey]
status: complete
updated: 2026-07-14
summary: "世界模型动作后果专题 · 03（3 篇）— RGB 之外补深度、光流与可仿真世界"
related:
  - ./robot-world-models-action-consequence-technology-map.md
  - ./wm-action-consequence-category-04-eval-posttrain.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
---

# 世界模型动作后果分类 03：3D/4D 几何与环境层

> **图谱分类节点**：对应 [具身智能研究室 · 世界模型动作后果专题](https://mp.weixin.qq.com/s/a5ZDDv70CLDfY98mfviWuA) 的 **03 3D/4D 几何与环境层** 段；总地图见 [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来观测与动作生成的具身策略 |
| VLA | Vision-Language-Action | 常被修正、筛选或后训练的上层策略 |
| WM | World Model | 预测动作后果的潜变量或视频模型 |
| MoE | Mixture of Experts | 异构动作模态的专家混合架构 |

## 本组工作

| 工作 | Wiki 实体 | 文内角色 |
|------|-----------|----------|
| RynnWorld-4D | [../entities/paper-rynnworld-4d-rgb-depth-flow](../entities/paper-rynnworld-4d-rgb-depth-flow.md) | 统一扩散同步预测 RGB、深度、光流 + Policy 头 |
| MECo-WAM | [../entities/paper-meco-wam-4d-geometry-cotraining](../entities/paper-meco-wam-4d-geometry-cotraining.md) | 训练期 4D 专家监督，推理移除保持轻量 WAM |
| EmbodiedGen V2 | [../entities/paper-embodiedgen-v2-sim-ready-world-engine](../entities/paper-embodiedgen-v2-sim-ready-world-engine.md) | 任务驱动可仿真 3D 世界引擎 |

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [动作后果技术地图](./robot-world-models-action-consequence-technology-map.md)

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
