---
type: method
tags: [vision, robotics, perception]
status: draft
updated: 2026-11-26
summary: "把视觉观测转换为机器人可使用的对象、空间或场景信息。"
sources:
  - ../../sources/courses/vision_transformer_primary_sources.md
related:
  - ../concepts/perception-coordinate-postprocessing.md
---

# 机器人视觉感知流程

## 一句话定义

把视觉观测转换为机器人可使用的对象、空间或场景信息。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| AI | Artificial Intelligence | 人工智能 |

## 关联概念

[相关概念](../concepts/perception-coordinate-postprocessing.md) 介绍该方法的背景。

## 主要技术路线

1. **观测与标定**：配置相机、时间同步、内外参和坐标系，记录遮挡与传感范围。
2. **视觉估计**：按任务选择检测、分割、深度或姿态估计；模型输出应明确坐标系及置信度。
3. **闭环验证**：将感知结果映射到规划控制接口，比较真实任务中的失败、延迟和环境变化。

[视觉表征学习](visual-representation-learning.md) 关注特征学习，本页关注传感到机器人决策的接口。

## 局限与后续核查

本页用于跨项目导航；具体数据、实现、评测和授权应以原始资料为准。

## 参考来源

- [资料归档](../../sources/courses/vision_transformer_primary_sources.md)
