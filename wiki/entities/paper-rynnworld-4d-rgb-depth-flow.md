---
type: entity
tags:
  - paper
  - world-models
  - world-action-models
  - depth
  - optical-flow
  - manipulation
  - alibaba
status: complete
updated: 2026-07-11
arxiv: "2607.06559"
related:
  - ../overview/wm-action-consequence-category-03-geometry-4d.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-meco-wam-4d-geometry-cotraining.md
  - ../entities/paper-embodiedgen-v2-sim-ready-world-engine.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "RynnWorld-4D（arXiv:2607.06559）：统一扩散同步生成 RGB、深度、光流；2.544 亿帧 Rynn4D 数据；RynnWorld-4D-Policy 单次前向读 4D 表征出动作，真机双手灵巧操作 SOTA 级。"
---

# RynnWorld-4D（4D Embodied World Models）

**RynnWorld-4D**（arXiv:2607.06559，[项目页](https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io)）——见策展导读与一手论文。

## 一句话定义

**从 2D 像素预测转向 RGB-DF 同步 4D 演化，Policy 头单次前向闭环出动作**——几何与运动与外观共同监督。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-DF | RGB, Depth, Flow | 外观-深度-光流三联表征 |
| DiT | Diffusion Transformer | 三分支跨模态注意力骨干 |
| IDM | Inverse Dynamics Model | RynnWorld-4D-Policy 逆动力学头 |

## 为什么重要

- 代表策展 **几何层** 推进：深度补结构、光流补跨帧运动。
- Policy 绕过多步视频去噪，降低 **世界预测→控制** 部署成本。
- 与 [MECo-WAM](../entities/paper-meco-wam-4d-geometry-cotraining.md) 对照：前者 **显式 4D 生成**，后者 **训练期几何专家、推理轻量**。

## 核心结构

| 模块 | 作用 |
|------|------|
| 三分支 DiT | RGB / 深度 / 光流互注意力同步演化 |
| 3D RoPE | 帧级几何-运动一致位置编码 |
| RynnWorld-4D-Policy | 读内部 4D latent 一次前向出机器人动作 |

## 实验要点（策展口径）

Rynn4D 2.544 亿帧；真机双手灵巧操作任务领先（见论文 Table）。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-03-geometry-4d](../overview/wm-action-consequence-category-03-geometry-4d.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.06559](https://arxiv.org/abs/2607.06559) — 一手论文
- [https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io](https://alibaba-damo-academy.github.io/RynnWorld-4D.github.io) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.06559](https://arxiv.org/abs/2607.06559)
