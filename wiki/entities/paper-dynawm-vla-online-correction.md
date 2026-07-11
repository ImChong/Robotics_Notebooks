---
type: entity
tags:
  - paper
  - world-action-models
  - vla
  - flow-matching
  - manipulation
  - online-correction
status: complete
updated: 2026-07-11
arxiv: "2607.02604"
related:
  - ../overview/wm-action-consequence-category-01-wam-action-prediction.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-dswam-dual-system-wam.md
  - ../entities/paper-dreamsteer-vla-deployment-steering.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "DynaWM（arXiv:2607.02604）：冻结基础 VLA，以多视角视觉历史与本体状态为条件，用 Mamba-3 动作编码 + V-JEPA 2.1 视觉编码 + 流匹配 DiT 在线重生成动作块，面向移动目标在线修正。"
---

# DynaWM（Dynamic World Model for VLA Action Correction）

**DynaWM**（arXiv:2607.02604，[项目页](https://arxiv.org/abs/2607.02604)）——见策展导读与一手论文。

## 一句话定义

**冻结 VLA 出初稿，世界模型据连续视觉与本体状态在线改写动作轨迹**——针对移动目标的速度/方向估计。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 在线动作修正模块 |
| VLA | Vision-Language-Action | 冻结的基础策略 |
| V-JEPA | Video Joint Embedding Predictive Architecture | 视觉历史编码器 |

## 为什么重要

- 单帧 VLA 难估计运动目标的速度方向；**连续视觉历史** 是世界模型在线修正的关键条件。
- 代表 WAM **第二类职责：修正已有动作** 而非从零生成。
- 与 [DSWAM](../entities/paper-dswam-dual-system-wam.md)（直接执行）、[DreamSteer](../entities/paper-dreamsteer-vla-deployment-steering.md)（筛选）互补。

## 核心结构

| 模块 | 作用 |
|------|------|
| 冻结 VLA | 输出初始动作块 |
| Mamba-3 动作编码器 | 组织动作历史条件 |
| V-JEPA 2.1 + 本体编码 | 多视角视觉与状态条件 |
| 流匹配 DiT | 重生成修正后动作轨迹 |

## 实验要点（策展口径）

策展文强调移动目标在线修正设定；具体 benchmark 以论文为准。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-01-wam-action-prediction](../overview/wm-action-consequence-category-01-wam-action-prediction.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.02604](https://arxiv.org/abs/2607.02604) — 一手论文
- [https://arxiv.org/abs/2607.02604](https://arxiv.org/abs/2607.02604) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.02604](https://arxiv.org/abs/2607.02604)
