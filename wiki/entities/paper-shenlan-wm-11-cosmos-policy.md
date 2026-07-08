---

type: entity
tags: [paper, world-models, shenlan-survey, open-source, nvidia, stanford]
status: complete
updated: 2026-07-08
arxiv: "2601.16163"
venue: —
summary: "微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。"
related:
  - ../overview/world-models-15-open-source-technology-map.md
  - ../overview/world-models-route-02-joint.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
  - ./paper-navwam-goal-conditioned-visual-navigation-wam.md
sources:
  - ../../sources/papers/shenlan_wm_survey_11_cosmos-policy.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
---

# Cosmos Policy

**Cosmos Policy** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 11/15** 篇，归类为 **02 联合架构**。

## 一句话定义

微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| VLA | Vision-Language-Action | 视觉-语言-动作端到端策略模型 |
| RGB-D | RGB + Depth | 彩色图与深度图联合感知 |

## 为什么重要

- 微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 02**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 11/15 |
| 路线 | 02 联合架构 |
| 出处 | — |
| 文内引用 | 61（2026-06-02，策展） |
| arXiv | [2601.16163](https://arxiv.org/abs/2601.16163) |

## 核心机制（归纳）

### 1）策展导读要点

未来观测与动作在同一扩散/自回归骨干中联合建模，减少级联误差累积；适合端到端 VLA/操控闭环。

### 2）策展导读要点

微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-02-joint.md](../overview/world-models-route-02-joint.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_11_cosmos-policy.md](../../sources/papers/shenlan_wm_survey_11_cosmos-policy.md)

## 参考来源

- [shenlan_wm_survey_11_cosmos-policy.md](../../sources/papers/shenlan_wm_survey_11_cosmos-policy.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 推荐继续阅读

- [arXiv:2601.16163](https://arxiv.org/abs/2601.16163) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)
