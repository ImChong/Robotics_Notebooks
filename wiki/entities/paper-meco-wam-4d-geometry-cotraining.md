---
type: entity
tags:
  - paper
  - world-action-models
  - 4d-geometry
  - vggt
  - manipulation
  - inference-efficient
status: complete
updated: 2026-07-11
arxiv: "2607.05468"
related:
  - ../overview/wm-action-consequence-category-03-geometry-4d.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-rynnworld-4d-rgb-depth-flow.md
  - ../entities/paper-dswam-dual-system-wam.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "MECo-WAM（arXiv:2607.05468）：训练期轻量 4D 专家 + 冻结 VGGT 几何监督、衰减 4D read-mask 与动作感知时序几何蒸馏；推理移除全部 4D 组件；LIBERO 98.2%、RoboTwin 2.0 92.6%。"
---

# MECo-WAM（Multi-Expert Co-Training WAM）

**MECo-WAM**（arXiv:2607.05468，[项目页](https://meco-wam.github.io/)）——见策展导读与一手论文。

## 一句话定义

**4D 几何只在训练期校正 video-action 表征，部署图与轻量 WAM 相同**——推理零额外几何开销。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 视频-动作联合策略 |
| VGGT | Visual Geometry Grounded Transformer | 冻结几何编码监督源 |
| MoT | Mixture-of-Transformers | 视频/动作/4D 多专家共训 |

## 为什么重要

- 回应「几何监督 vs 推理延迟」矛盾——**训练加料、部署不加重**。
- 与 RynnWorld-4D **显式 4D 生成** 形成设计光谱两端。
- LIBERO 98.2%、RoboTwin 92.6% 说明几何先验可转化为操纵增益。

## 核心结构

| 机制 | 作用 |
|------|------|
| 4D 专家 | 冻结 VGGT 关系目标监督 |
| 衰减 read-mask | 早期允许读几何，后期撤依赖防捷径 |
| 时序几何蒸馏 | 对齐帧内/跨帧几何与动作相关区域 |
| 部署 | 移除全部 4D 辅助模块 |

## 实验要点（策展口径）

LIBERO 98.2%；RoboTwin 2.0 92.6%；真机挑战任务提升（见论文）。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-03-geometry-4d](../overview/wm-action-consequence-category-03-geometry-4d.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.05468](https://arxiv.org/abs/2607.05468) — 一手论文
- [https://meco-wam.github.io/](https://meco-wam.github.io/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.05468](https://arxiv.org/abs/2607.05468)
