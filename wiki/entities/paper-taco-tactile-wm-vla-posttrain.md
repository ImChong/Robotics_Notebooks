---
type: entity
tags:
  - paper
  - world-models
  - tactile
  - vla
  - post-training
  - contact-rich
  - manipulation
status: complete
updated: 2026-07-11
arxiv: "2607.02840"
related:
  - ../overview/wm-action-consequence-category-02-contact-modeling.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ../entities/paper-vt-wam-visuotactile-contact-rich.md
  - ../entities/paper-dreamsteer-vla-deployment-steering.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md
summary: "TACO（arXiv:2607.02840）：识别-想象-标注闭环——进度动作模型定位失败邻域，视触觉生成模型补修正片段并标注动作；知识隔离触觉适应 + 优势条件训练；相对基础策略 +44pp。"
---

# TACO（TActile World Model as a Self-COrrector）

**TACO**（arXiv:2607.02840，[项目页](https://taco-wm.github.io/)）——见策展导读与一手论文。

## 一句话定义

**把真实失败轨迹转为想象纠错片段，规模化 VLA 接触后训练**——补齐成功示范稀缺的调整窗口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 被后训练的基础策略 |
| WM | World Model | 视触觉生成修正片段 |
| RL | Reinforcement Learning | 优势条件训练接口 |

## 为什么重要

- 长任务缺 **接触偏掉后的恢复记录**；TACO 把失败当数据矿。
- 相对基础策略 **+44pp** 绝对成功率。
- 与 [DreamSteer](../entities/paper-dreamsteer-vla-deployment-steering.md) 形成 **后训练 vs 部署筛选** 对照。

## 核心结构

| 阶段 | 作用 |
|------|------|
| Recognize | 进度动作模型定位失败邻域状态 |
| Imagine | 视触觉 WM 生成局部修正片段 |
| Label | 为片段标注可执行纠错动作 |
| Adapt | 知识隔离触觉适应，保护 VLM 先验 |

## 实验要点（策展口径）

真机接触任务：相对基础策略 +44pp；无知识隔离 +32pp 增益差距。

## 常见误区或局限

- 策展文强调的问题：**动作忠实度、长时序误差、不确定性、跨本体接口** 对本工作仍适用；细节局限以论文讨论为准。
- 公众号数字为 **导读归纳**，复现实验请核对 arXiv 与项目页。

## 与其他页面的关系

- 分类 hub：[wm-action-consequence-category-02-contact-modeling](../overview/wm-action-consequence-category-02-contact-modeling.md)
- 父地图：[动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- 概念对照：[World Action Models](../concepts/world-action-models.md)

## 推荐继续阅读

- [arXiv:2607.02840](https://arxiv.org/abs/2607.02840) — 一手论文
- [https://taco-wm.github.io/](https://taco-wm.github.io/) — 项目页与演示

## 参考来源

- [wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md](../../sources/blogs/wechat_embodied_ai_lab_robot_world_models_action_consequence_2026.md)
- [arXiv:2607.02840](https://arxiv.org/abs/2607.02840)
