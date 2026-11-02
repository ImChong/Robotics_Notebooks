---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-visual-affordance-prediction-survey-and-reproducibility
arxiv: "2505.05074"
venue: "arXiv 2025"
summary: "arXiv 2025 的 affordance 论文，清单分组：Related Surveys（Focus: Visual affordance prediction with an emphasis on reproducibility）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_visual-affordance-prediction-survey-and-reproducibility.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Visual Affordance Prediction

**Visual Affordance Prediction: Survey and Reproducibility**（arXiv 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Visual Affordance Prediction: Survey and Reproducibility——在清单中归入 **Related Surveys（Focus: Visual affordance prediction with an emphasis on reproducibility）**（清单未附方法摘要；论文的组织结构见「流程总览」，贡献概括以原文为准）。

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

下图是论文自己的章节框架与统一形式化（综述组织结构，非本页自创分类）。

```mermaid
flowchart TD
    DEF["§II 统一形式化<br/>f(x_v, T, e) → {a, o, S, P}<br/>what: 动作 a 与物体 o / where: 交互区域 S / how: 手部位姿 P"] --> LOC
    subgraph REV["§III 按子任务综述方法"]
        LOC["A. Object localisation<br/>定位任务相关物体"] --> CLS["B. Functional classification<br/>预测每个物体上的动作"]
        CLS --> SEG["C. Functional segmentation<br/>affordance detection and segmentation / affordance grounding"]
        SEG --> HAND["D. Hand pose estimation and synthesis<br/>估计手部位姿并渲染到图像"]
    end
    HAND --> DS["§IV Datasets: review and limitations<br/>图像数据集对比与标注局限"]
    DS --> REP["§V Affordance prediction reproducibility<br/>A. 可复现性挑战 RC1-RC5<br/>B. Affordance Sheets"]
    REP --> SHEET["Affordance Sheet 六部分<br/>问题 / 数据集 RC1 / 模型 RC2-3 / 实验设置 RC4 / 性能度量 RC5 / 机器人验证"]
    SHEET --> FUT["§VI Future directions<br/>物体物理属性 / AI agent 集成 / 扩大数据集规模 / 基准评测"]
```

图依据：综述组织结构图（章节框架），论文 §II（式 1）、§III 开头的五个子任务列表与 §III-A 至 §III-D、§IV、§V-A/B（Table IX 的六部分 Affordance Sheet）、§VI。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2025 |
| 方法名（清单标注） | - |
| 分组 1 | Related Surveys · Focus: Visual affordance prediction with an emphasis on reproducibility |
| 论文 | <https://arxiv.org/abs/2505.05074> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；论文的组织结构见上文「流程总览」（依据论文各章节，未逐页核对实验数字），其余细节与实验请读 [原文](https://arxiv.org/abs/2505.05074)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2505.05074) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Visual Affordance Prediction」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述的论文的组织结构（依据论文各章节，未逐页核对实验数字）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_visual-affordance-prediction-survey-and-reproducibility.md`](../../sources/papers/affordance_awesome_visual-affordance-prediction-survey-and-reproducibility.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2505.05074>
- [arXiv PDF](https://arxiv.org/pdf/2505.05074)：已通读 §I 引言、§II 问题形式化、§V 可复现性与 Affordance Sheet、§VI 未来方向；§III 各子任务与 §IV 数据集读各节开头与总述（方法 / 数据集对比表略读）；附录 Affordance Sheet 表未逐项读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2505.05074)
