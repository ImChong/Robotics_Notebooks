---
type: entity
tags:
  - paper
  - world-models
  - simulation
  - 3d-generation
  - embodied-ai
  - horizon
status: complete
updated: 2026-07-11
arxiv: "2607.07459"
related:
  - ../overview/wm-action-consequence-category-03-geometry-4d.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
  - ../entities/paper-deform360-deformable-visuotactile-dataset.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "EmbodiedGen V2（arXiv:2607.07459）：生成带碰撞体、物理属性、交互可供性与模拟器接口的 3D 任务世界；83.3% 任务世界免改可用；在线 RL 仿真 9.7%→79.8%、真机 21.7%→75.0%。"
---

# EmbodiedGen V2（Simulation-Ready 3D World Engine）

**EmbodiedGen V2**（arXiv:2607.07459，[项目页](https://horizonrobotics.github.io/EmbodiedGen/)）——见策展导读与一手论文。

## 一句话定义

**从可看场景到可执行环境：任务驱动生成 sim-ready 3D 世界，直连导航/操作/RL 闭环**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 生成环境上在线强化学习 |
| Sim | Simulation | 跨模拟器可部署资产与世界 |
| WM | World Model | 环境层世界生成引擎 |

## 为什么重要

- 补齐策展 **环境层**：碰撞、物理、任务语义、模拟器 API。
- 83.3% 任务世界无需人工修改即可下游仿真。
- 与像素/几何 WM 互补：**可训练可评测的完整场景** 而非单帧未来视频。

## 核心结构

| 能力 | 说明 |
|------|------|
| 资产管线 | 96.5% 人工接受率、98.6% 碰撞成功率 |
| 任务世界 | 操作/导航/移动操作、多房间大场景 |
| 策略训练 | 生成环境在线 RL 大幅提升成功率 |

## 实验要点（策展口径）

83.3% 任务世界免改可用；RL 仿真 9.7%→79.8%；真机 21.7%→75.0%。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-03-geometry-4d](../overview/wm-action-consequence-category-03-geometry-4d.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.07459](https://arxiv.org/abs/2607.07459) — 一手论文
- [https://horizonrobotics.github.io/EmbodiedGen/](https://horizonrobotics.github.io/EmbodiedGen/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.07459](https://arxiv.org/abs/2607.07459)
