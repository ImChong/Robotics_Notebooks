---
type: entity
tags:
  - paper
  - world-models
  - moe
  - diffusion-transformer
  - heterogeneous-control
  - tsinghua
  - manifold
status: complete
updated: 2026-07-11
arxiv: "2607.03964"
related:
  - ../overview/wm-action-consequence-category-01-wam-action-prediction.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "Worldscape-MoE（arXiv:2607.03964）：DiT 上共享专家学通用场景动力学 + 控制专属专家处理相机/关节/手部异构动作；渐进 MoE 微调扩展新模态；WorldArena 强结果。"
---

# Worldscape-MoE（Unified Mixture-of-Experts World Model）

**Worldscape-MoE**（arXiv:2607.03964，[项目页](https://worldscape-moe.com)）——见策展导读与一手论文。

## 一句话定义

**异构动作接口约束同一物理世界：共享专家学规律，专属专家保各模态控制精度**——MoE 扩展世界模型动作空间。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoE | Mixture of Experts | 共享+模态专属专家 |
| DiT | Diffusion Transformer | 视频世界模型骨干 |
| WM | World Model | 可扩展异构动作条件的生成式模拟器 |

## 为什么重要

- 回答 **跨本体/跨模态动作接口** 如何共享世界规律（策展文开放问题之一）。
- 相机运动、机器人关节、第一视角手部控制 **表征完全不同但约束同一物理**。
- 与 [GigaWorld-1](../entities/paper-gigaworld-1-policy-evaluation.md) 同属 **规模化世界模型基础设施** 方向。

## 核心结构

| 组件 | 作用 |
|------|------|
| 共享专家 | 积累通用场景动力学 |
| 控制专属专家 | 各动作模态接口 |
| 渐进 MoE 微调 | 持续加入新动作类型 |

## 实验要点（策展口径）

locomotion、操作、egocentric 手部控制；WorldArena 榜单强表现（见论文）。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-01-wam-action-prediction](../overview/wm-action-consequence-category-01-wam-action-prediction.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.03964](https://arxiv.org/abs/2607.03964) — 一手论文
- [https://worldscape-moe.com](https://worldscape-moe.com) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.03964](https://arxiv.org/abs/2607.03964)
