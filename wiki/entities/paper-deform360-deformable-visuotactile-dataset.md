---
type: entity
tags:
  - paper
  - dataset
  - deformable-objects
  - tactile
  - world-models
  - visuotactile
status: complete
updated: 2026-07-11
arxiv: "2607.05390"
related:
  - ../overview/wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-rynnworld-4d-rgb-depth-flow.md
  - ../entities/paper-vt-wam-visuotactile-contact-rich.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "Deform360（arXiv:2607.05390）：198 类日常可变形物体、1980 序列、215+ 小时；41 环绕相机 + 双手触觉夹爪；系统比较 2D 视频 WM 与 3D 粒子模型并演示可变形体规划。"
---

# Deform360（Massive Multi-view Visuotactile Dataset）

**Deform360**（arXiv:2607.05390，[项目页](https://deform360.lhy.xyz)）——见策展导读与一手论文。

## 一句话定义

**大规模视触觉可变形物体数据：全局运动 + 局部接触形变，支撑 2D/3D 世界模型对照评测**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 2D 视频 vs 3D 粒子动力学对照对象 |
| MPC | Model Predictive Control | 数据集上初步规划演示 |
| RGB | Red-Green-Blue | 多视角外观记录 |

## 为什么重要

- 可变形体高维状态使 **纯 2D 视频 WM** 与 **显式 3D 几何 WM** 优劣难定论；Deform360 提供对照基准。
- 补齐接触专题中的 **数据层**（与 VT-WAM/TACO 方法层互补）。
- 215+ 小时规模支持结构先验 vs 可扩展性权衡分析。

## 核心结构

| 组成 | 规模 |
|------|------|
| 物体类别 | 198 类日常可变形物体 |
| 交互序列 | 1980 条 |
| 采集 | 41 环绕相机 + 双手触觉夹爪 |
| 标注 | 无标记视触觉 3D 跟踪管线 |

## 实验要点（策展口径）

198 物体 / 1980 序列 / 215+ 小时；2D vs 3D WM 系统评测（见论文）。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-02-contact-modeling](../overview/wm-action-consequence-category-02-contact-modeling.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.05390](https://arxiv.org/abs/2607.05390) — 一手论文
- [https://deform360.lhy.xyz](https://deform360.lhy.xyz) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.05390](https://arxiv.org/abs/2607.05390)
