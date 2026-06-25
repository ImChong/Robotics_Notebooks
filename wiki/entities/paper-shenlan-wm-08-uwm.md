---

type: entity
tags: [paper, world-models, shenlan-survey, open-source, toyota-research, uw]
status: complete
updated: 2026-06-25
arxiv: "2504.02792"
venue: RSS 2025
summary: "视频与动作扩散统一 Transformer，模态特定扩散时间步表策略/动力学/生成器。"
related:
  - ../overview/world-models-15-open-source-technology-map.md
  - ../overview/world-models-route-02-joint.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/shenlan_wm_survey_08_uwm.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
---

# Unified World Models (UWM)

**Unified World Models (UWM)** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 08/15** 篇，归类为 **02 联合架构**。

## 一句话定义

视频与动作扩散统一 Transformer，模态特定扩散时间步表策略/动力学/生成器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UWM | Unified World Models | 统一世界模型，联合建模观测与动作 |

## 为什么重要

- 视频与动作扩散统一 Transformer，模态特定扩散时间步表策略/动力学/生成器。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 02**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 08/15 |
| 路线 | 02 联合架构 |
| 出处 | RSS 2025 |
| 文内引用 | 102（2026-06-02，策展） |
| arXiv | [2504.02792](https://arxiv.org/abs/2504.02792) |

## 核心机制（归纳）

### 1）策展导读要点

未来观测与动作在同一扩散/自回归骨干中联合建模，减少级联误差累积；适合端到端 VLA/操控闭环。

### 2）策展导读要点

视频与动作扩散统一 Transformer，模态特定扩散时间步表策略/动力学/生成器。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-02-joint.md](../overview/world-models-route-02-joint.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_08_uwm.md](../../sources/papers/shenlan_wm_survey_08_uwm.md)

## 参考来源

- [shenlan_wm_survey_08_uwm.md](../../sources/papers/shenlan_wm_survey_08_uwm.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 推荐继续阅读

- [arXiv:2504.02792](https://arxiv.org/abs/2504.02792) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)
