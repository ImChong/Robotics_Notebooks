---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-one-shot-transfer-of-affordance-regions-affcorrs
arxiv: "2209.07147"
venue: "CoRL 2022"
summary: "CoRL 2022 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_one-shot-transfer-of-affordance-regions-affcorrs.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffCorrs

**One-Shot Transfer of Affordance Regions? AffCorrs!**（CoRL 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

One-Shot Transfer of Affordance Regions? AffCorrs!——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
    SUP["支持图像 IR + 查询部件掩码 MQ"] --> DESC["DINO-ViT 提取稠密描述子 FR、FT"]
    TGT["目标图像 IT（目标掩码 MT 由显著性或用户给出）"] --> DESC
    DESC --> SEL["Step1：用 MQ 与 MT 只保留前景描述子"]
    SEL --> CLU["Step2：K-Means 聚类为 KQ 个查询质心与 KT 个目标质心"]
    CLU --> FWD["前向匹配：质心间余弦相似度 + softmax，求和得投票 VQT"]
    CLU --> BWD["后向匹配：目标质心对完整支持描述子 FR 匹配，得落入查询区域的概率 PTQ"]
    FWD --> SC["Step3：得分 S = VQT · PTQ，映射回各质心所含像素"]
    BWD --> SC
    SC --> CRF["CRF（前景能量为得分，背景为常数阈值）→ 平滑二值掩码"]
    CRF --> OUT["目标物体上的对应 affordance 区域"]
    OUT --> SKILL["可选：把与该部件绑定的技能 Si 迁移到目标部件执行"]
```

图依据：论文 方法部分（Pre-processing、Forward/Backward matching、Post-processing）/ Fig.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CoRL 2022 |
| 方法名（清单标注） | AffCorrs |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Generalizable |
| 论文 | <https://arxiv.org/abs/2209.07147> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2209.07147)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2209.07147) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffCorrs」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_one-shot-transfer-of-affordance-regions-affcorrs.md`](../../sources/papers/affordance_awesome_one-shot-transfer-of-affordance-regions-affcorrs.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2209.07147>
- [arXiv PDF](https://arxiv.org/pdf/2209.07147)：已通读方法部分（实验与附录以略读为准）：方法部分（问题定义、预处理、前向/后向匹配、后处理、设计选择、单次技能迁移）；实验与附录略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2209.07147)
