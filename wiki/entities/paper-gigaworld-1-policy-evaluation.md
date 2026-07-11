---
type: entity
tags:
  - paper
  - world-models
  - policy-evaluation
  - benchmark
  - manipulation
  - simulation
status: complete
updated: 2026-07-11
arxiv: "2607.02642"
related:
  - ../overview/wm-action-consequence-category-04-eval-posttrain.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-dreamsteer-vla-deployment-steering.md
  - ../entities/paper-embodiedgen-v2-sim-ready-world-engine.md
  - ../entities/paper-worldscape-moe-heterogeneous-action.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "GigaWorld-1（arXiv:2607.02642）：系统研究 7 类视频 WM、4 种动作编码、32.4 万+ 模拟策略轨迹；结论：评估器质量取决于长时序动作忠实 rollout，而非短时视觉逼真；发布 WMBench 与 GigaWorld-1 模型。"
---

# GigaWorld-1（World Models for Robot Policy Evaluation）

**GigaWorld-1**（arXiv:2607.02642，[项目页](https://open-gigaai.github.io/giga-world-1/)）——见策展导读与一手论文。

## 一句话定义

**世界模型作策略评估器：短时画面逼真不够，长时序动作忠实与排序一致性才是关键**——真机评测瓶颈的系统研究。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 策略轨迹生成与排序的 surrogate 评估器 |
| WMBench | World Model Benchmark | 真机遥操作与策略 rollout 对照基准 |
| VLA | Vision-Language-Action | 被评估的具身基础策略家族 |

## 为什么重要

- 真机评测贵、慢；世界模型若 **动作不敏感** 则评估失效（策展文举例：差几厘米却都生成抓取成功视频）。
- 32.4 万轨迹 + 1.2 万小时预训练视频的系统实证。
- 与 [DreamSteer](../entities/paper-dreamsteer-vla-deployment-steering.md) 的 **候选筛选**、[EmbodiedGen V2](../entities/paper-embodiedgen-v2-sim-ready-world-engine.md) 的 **环境扩展** 同属研发链路入口。

## 核心结构

| 发现 | 含义 |
|------|------|
| 长时序动作忠实 | 比短时视觉逼真更决定评估质量 |
| 预训练配比 | 通用世界知识 vs 机器人可控性需平衡 |
| 架构选择 | 动作编码、记忆、评估导向后训练影响真机对齐 |

## 实验要点（策展口径）

7 类视频 WM；324k+ rollout；CVPR 2026 GigaBrain Challenge 社区提交纳入分析。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-04-eval-posttrain](../overview/wm-action-consequence-category-04-eval-posttrain.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.02642](https://arxiv.org/abs/2607.02642) — 一手论文
- [https://open-gigaai.github.io/giga-world-1/](https://open-gigaai.github.io/giga-world-1/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.02642](https://arxiv.org/abs/2607.02642)
