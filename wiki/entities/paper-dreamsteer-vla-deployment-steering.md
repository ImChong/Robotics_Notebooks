---
type: entity
tags:
  - paper
  - world-models
  - vla
  - deployment
  - model-based-rl
  - manipulation
status: complete
updated: 2026-07-11
arxiv: "2607.02865"
related:
  - ../overview/wm-action-consequence-category-01-wam-action-prediction.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-taco-tactile-wm-vla-posttrain.md
  - ../entities/paper-gigaworld-1-policy-evaluation.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "DreamSteer（arXiv:2607.02865）：部署时冻结 VLA 采样多候选动作块，潜变量世界模型预演未来观测，语言条件价值模型排序后执行；四组真机基准成功率 23.75%→66.25%。"
---

# DreamSteer（Latent World Model Steering for VLA）

**DreamSteer**（arXiv:2607.02865，[项目页](https://dream-steer.github.io/)）——见策展导读与一手论文。

## 一句话定义

**不改 VLA 参数：世界模型预演候选动作，价值模型在部署端筛选最优块**——零微调 steering。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 被 steering 的冻结策略 |
| WM | World Model | 动作条件潜变量未来预测 |
| LVM | Language-conditioned Value Model | 按指令排序想象轨迹 |

## 为什么重要

- **第三类 WAM 职责：部署前筛选**——与直接执行、在线修正并列。
- 不需目标环境微调数据，降低部署分布偏移成本。
- 与 [TACO](../entities/paper-taco-tactile-wm-vla-posttrain.md) 对比：DreamSteer 走 **推理时筛选**，TACO 走 **后训练纠错数据**。

## 核心结构

| 步骤 | 作用 |
|------|------|
| 候选采样 | 冻结 VLA + 预定义运动原语 |
| 想象 | 潜变量 WM 预测各候选未来观测 |
| 排序 | 语言价值模型按任务指令打分 |
| 执行 | 仅部署高分候选 |

## 实验要点（策展口径）

四组真机：成功率 23.75%→66.25%；指令遵循 38.75%→56.25%。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-01-wam-action-prediction](../overview/wm-action-consequence-category-01-wam-action-prediction.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.02865](https://arxiv.org/abs/2607.02865) — 一手论文
- [https://dream-steer.github.io/](https://dream-steer.github.io/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.02865](https://arxiv.org/abs/2607.02865)
