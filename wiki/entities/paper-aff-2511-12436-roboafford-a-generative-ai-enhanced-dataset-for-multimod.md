---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-roboafford-a-generative-ai-enhanced-dataset-for-multimod
arxiv: "2511.12436"
venue: "IROSW 2025"
summary: "IROSW 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image/language）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_roboafford-a-generative-ai-enhanced-dataset-for-multimod.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# RoboAfford++

**RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robotic Manipulation and Navigation**（IROSW 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robotic Manipulation and Navigation——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image/language）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart TB
    SRC["数据收集与过滤<br/>LVIS / Pixmo-Points / PACO-LVIS<br/>RoboPoint 的 Object & Region Reference<br/>AI2Thor 采集的 NaviAfford"]
    FIL["过滤：丢弃 &gt;10 点标注<br/>GPT-4o 保留室内物体"]
    SRC --> FIL
    FIL --> T1["人工设计模板（28 个）<br/>物体 affordance 识别 QA"]
    FIL --> T2["GPT-4o 提示生成<br/>物体 affordance 预测 QA<br/>框 + mask 采样点"]
    FIL --> T3["改写 RoboPoint 标注<br/>空间 affordance 定位 QA<br/>每题最多 10 点"]
    T1 --> DS["RoboAfford++ 数据集<br/>869,987 图 / 2.0M QA"]
    T2 --> DS
    T3 --> DS
    DS --> S1["阶段 1：General Localization Learning<br/>LVIS + Pixmo-Points"]
    S1 --> S2["阶段 2：Object-Spatial Affordance Enhancement<br/>其余四个数据源"]
    S2 --> M["RoboAfford-Qwen++<br/>视觉编码器 + MLP 投影 + Qwen2.5 LLM"]
    M --> P["输出 2D affordance 点"]
    P --> D["深度图转 3D 坐标<br/>或作为分割提示"]
    D --> E["机器人执行<br/>抓取位姿生成 / 导航目标"]
```

图依据：论文 §III-A/B（数据收集与 QA 生成，Fig. 2）、§IV（Methodology，Fig. 3）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROSW 2025 |
| 方法名（清单标注） | RoboAfford++ |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: image/language |
| 论文 | <https://arxiv.org/abs/2511.12436> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2511.12436)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2511.12436) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「RoboAfford++」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/affordance_awesome_roboafford-a-generative-ai-enhanced-dataset-for-multimod.md`](../../sources/papers/affordance_awesome_roboafford-a-generative-ai-enhanced-dataset-for-multimod.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2511.12436>
- [arXiv PDF](https://arxiv.org/pdf/2511.12436)：已通读方法部分（实验与附录以略读为准）：§III 数据集与基准构建、§IV Methodology、§V-A 实验设置

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2511.12436)
