---
type: entity
tags:
  - paper
  - world-action-models
  - flow-matching
  - manipulation
  - deformable-objects
status: complete
updated: 2026-07-11
arxiv: "2607.04927"
related:
  - ../overview/wm-action-consequence-category-01-wam-action-prediction.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-dynawm-vla-online-correction.md
  - ../entities/paper-dreamsteer-vla-deployment-steering.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "DSWAM（arXiv:2607.04927）：双系统 WAM——System 1 世界动作执行器视频协同训练、推理直出动作块；可选 System 2 VLM 规划器拆分子任务；真机折叠 96.3%、RoboTwin 2.0 92.38%。"
---

# DSWAM（Dual-System World Action Foundation Model）

**DSWAM**（arXiv:2607.04927，[项目页](https://ds-wam.github.io/)）——见策展导读与一手论文。

## 一句话定义

**双系统 WAM：执行器学物理变化并直出动作块，粗粒度指令才由 VLM 规划器拆解**——训练学未来、部署不等视频去噪。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 联合未来与动作 |
| VLA | Vision-Language-Action | System 2 可选规划器 |
| TRT | TensorRT | 真机低延迟推理加速 |

## 为什么重要

- **训练–部署解耦范例：** 视频协同训练提供时序监督，推理接口仍为动作块 + TensorRT/异步分块。
- **可变形折叠强实证：** 相同协议下成功率 92.5%→96.3%，用时 2:18→1:44。
- 与 [DynaWM](../entities/paper-dynawm-vla-online-correction.md)、[DreamSteer](../entities/paper-dreamsteer-vla-deployment-steering.md) 形成 WAM **执行 / 修正 / 筛选** 三角。

## 核心结构

| 模块 | 作用 |
|------|------|
| System 1 执行器 | 动作预测 + 视频协同训练；真机直出动作块 |
| System 2 规划器 | 粗粒度多步指令时拆分子任务 |
| 部署栈 | TensorRT、异步执行、实时动作分块 |

## 实验要点（策展口径）

真机折叠 96.3%；RoboTwin 2.0 clean 92.38%、randomized 91.90%。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-01-wam-action-prediction](../overview/wm-action-consequence-category-01-wam-action-prediction.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.04927](https://arxiv.org/abs/2607.04927) — 一手论文
- [https://ds-wam.github.io/](https://ds-wam.github.io/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.04927](https://arxiv.org/abs/2607.04927)
