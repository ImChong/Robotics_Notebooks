---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2405.19917"
venue: "ECCV 2024"
summary: "Systematically studies the cross-domain few-shot scenario of egocentric action recognition for the first time, proposing a multimodal distillation framework"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2405_19917_mm-cdfsl-multimodal-cross-domain-few-sho.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# MM-CDFSL

**MM-CDFSL: Multimodal Cross-Domain Few-Shot Learning for Egocentric Action Recognition** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 168/249** 篇，分组 **Action Recognition & Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Systematically studies the cross-domain few-shot scenario of egocentric action recognition for the first time, proposing a multimodal distillation framework

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Systematically studies the cross-domain few-shot scenario of egocentric action recognition for the first time, proposing a multimodal distillation framework
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["源域标注数据 + 目标域无标注数据（RGB / 光流 / 手部姿态热图）"] --> B["I 预训练：每个模态独立训练 VideoMAE（Tube Masking）"]
    B --> C["联合目标：源+目标域重建 L2 + 源域动作分类交叉熵"]
    C --> D["II 多模态蒸馏：冻结各模态教师编码器"]
    D --> E["学生 RGB 编码器 + 投影 MLP，对齐教师特征（L2，stop-gradient）"]
    E --> F["III 少样本适应：保留 RGB 学生编码器，用 N-way K-shot 支持集训练分类头 G'"]
    F --> G["IV Ensemble Masked Inference：对查询样本施加 P 种 Tube Mask，平均 softmax 概率"]
    G --> H["输出动作类别"]
```

图依据：论文 §3.1–3.4（Fig. 2）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 168/249 |
| 分组 | Action Recognition & Detection |
| 出处 | ECCV 2024 |
| 论文 | <https://arxiv.org/abs/2405.19917> |

## 核心机制（归纳）

### 策展导读要点

Systematically studies the cross-domain few-shot scenario of egocentric action recognition for the first time, proposing a multimodal distillation framework

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2405.19917) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Action Recognition & Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MM-CDFSL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action Recognition & Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2405_19917_mm-cdfsl-multimodal-cross-domain-few-sho.md`](../../sources/papers/sun_awesome_ego_2405_19917_mm-cdfsl-multimodal-cross-domain-few-sho.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2405.19917>
- [arXiv PDF](https://arxiv.org/pdf/2405.19917)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 方法全部；§4 实验与附录略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2405.19917)
