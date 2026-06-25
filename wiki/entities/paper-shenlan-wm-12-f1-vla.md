---

type: entity
tags: [paper, world-models, shenlan-survey, open-source, hit, shanghai-ai-lab]
status: complete
updated: 2026-06-25
arxiv: "2509.06951"
venue: —
summary: "MoE 三模块：感知、视觉预见、控制；下一尺度预测 + 预见引导 IDM。"
related:
  - ../overview/world-models-15-open-source-technology-map.md
  - ../overview/world-models-route-02-joint.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/shenlan_wm_survey_12_f1-vla.md
  - ../../sources/papers/shenlan_world_models_15_reference_catalog.md
  - ../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md
---

# F1-VLA

**F1-VLA** 收录于 [深蓝具身智能 · 世界模型 15 开源项目专题](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg) **第 12/15** 篇，归类为 **02 联合架构**。

## 一句话定义

MoE 三模块：感知、视觉预见、控制；下一尺度预测 + 预见引导 IDM。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoE | Mixture-of-Experts | 门控网络加权组合多个专家子网络 |
| IDM | Inverse Dynamics Model | 由（未来）状态反推所需动作/力矩的模型 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |

## 为什么重要

- MoE 三模块：感知、视觉预见、控制；下一尺度预测 + 预见引导 IDM。
- 属于 [世界模型 15 项目地图](../overview/world-models-15-open-source-technology-map.md) **路线 02**。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 12/15 |
| 路线 | 02 联合架构 |
| 出处 | — |
| 文内引用 | 24（2026-06-02，策展） |
| arXiv | [2509.06951](https://arxiv.org/abs/2509.06951) |

## 核心机制（归纳）

### 1）策展导读要点

未来观测与动作在同一扩散/自回归骨干中联合建模，减少级联误差累积；适合端到端 VLA/操控闭环。

### 2）策展导读要点

MoE 三模块：感知、视觉预见、控制；下一尺度预测 + 预见引导 IDM。

## 常见误区

1. 开源 WM 项目的引用量与 **控制一致性/下游任务增益** 无简单线性关系；复现前需核对 License 与权重。

## 实验与评测

- 本页在公众号/survey **策展编译**基础上补充机制归纳；**量化 benchmark、消融与实机指标以原文 PDF / 项目页为准**（链接见 [参考来源](#参考来源)）。
- 与同栈姊妹篇对照时，请回到对应 **技术地图 / 42 篇栈 / BFM 地图 / VLN 地图** 总览中的实验段落。

## 与其他页面的关系

- 路线 hub：[world-models-route-02-joint.md](../overview/world-models-route-02-joint.md)
- 总地图：[world-models-15-open-source-technology-map.md](../overview/world-models-15-open-source-technology-map.md)
- 原始 source：[shenlan_wm_survey_12_f1-vla.md](../../sources/papers/shenlan_wm_survey_12_f1-vla.md)

## 参考来源

- [shenlan_wm_survey_12_f1-vla.md](../../sources/papers/shenlan_wm_survey_12_f1-vla.md)
- [shenlan_world_models_15_reference_catalog.md](../../sources/papers/shenlan_world_models_15_reference_catalog.md)
- [wechat_shenlan_world_models_15_open_source_2026.md](../../sources/blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 推荐继续阅读

- [arXiv:2509.06951](https://arxiv.org/abs/2509.06951) — 论文全文
- [深蓝具身智能原文](https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg)
