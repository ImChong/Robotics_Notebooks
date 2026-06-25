---

type: entity
tags: [paper, world-models, shenlan-survey, open-source, google-deepmind, nyu, stanford]
status: complete
updated: 2026-06-25
arxiv: "2506.00613"
venue: —
summary: "WM 作策略评估虚拟靶场；VLM 奖励 + MC 模拟，与真机排名高度相关。"
related:
  - ../overview/world-models-15-open-source-technology-map.md
  - ../overview/world-models-route-03-virtual-sandbox.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/shenlan_wm_survey_15_worldgym.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
---

# WorldGym

**WorldGym** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 15/15** 篇，归类为 **03 虚拟沙盒**。

## 一句话定义

WM 作策略评估虚拟靶场；VLM 奖励 + MC 模拟，与真机排名高度相关。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |

## 为什么重要

- WM 作策略评估虚拟靶场；VLM 奖励 + MC 模拟，与真机排名高度相关。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 03**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 15/15 |
| 路线 | 03 虚拟沙盒 |
| 出处 | — |
| 文内引用 | 13（2026-06-02，策展） |
| arXiv | [2506.00613](https://arxiv.org/abs/2506.00613) |

## 核心机制（归纳）

### 1）策展导读要点

世界模型作为 RL/评估虚拟环境，在想象中 rollout 替代昂贵真机试错；强调物理一致性与下游策略增益。

### 2）策展导读要点

WM 作策略评估虚拟靶场；VLM 奖励 + MC 模拟，与真机排名高度相关。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-03-virtual-sandbox.md](../overview/world-models-route-03-virtual-sandbox.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_15_worldgym.md](../../sources/papers/shenlan_wm_survey_15_worldgym.md)

## 参考来源

- [shenlan_wm_survey_15_worldgym.md](../../sources/papers/shenlan_wm_survey_15_worldgym.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 推荐继续阅读

- [arXiv:2506.00613](https://arxiv.org/abs/2506.00613) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)
