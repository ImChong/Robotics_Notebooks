---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-foresightful-dense-visual-affordance-for-deform
arxiv: "2303.11057"
venue: "ICCV 2023"
summary: "ICCV 2023 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_learning-foresightful-dense-visual-affordance-for-deform.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# DefoAfford

**Learning Foresightful Dense Visual Affordance for Deformable Object Manipulation**（ICCV 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning Foresightful Dense Visual Affordance for Deformable Object Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph DC["Fold to Unfold 多阶段数据采集（由终态向复杂态）"]
        D1["阶段 i 的状态 o_i：选 p_pick、p_place 得 o_i+1"] --> D2["执行反向动作（拾取 p_place、放到 p_pick）得 o_i'"]
        D2 --> D3{"o_i' 与 o_i 相似？"}
        D3 -- 是 --> D4["o_i+1 作为阶段 i+1 的起始状态，并采样多样动作及其结果"]
    end
    subgraph TRN["逐阶段监督学习（与推理顺序相反）"]
        T1["第 1 阶段：以直接距离 1 - dist(o', T) 监督放置 affordance"] --> T2["第 i 阶段：以 α·value + β·(1 - dist) 监督，value 来自上一阶段已训练的 M_pick"]
        T2 --> T3["每阶段先训练 M_place，再用其最大放置 affordance 监督 M_pick"]
        T3 --> T4["集成系统训练 IST：以 M_pick + M_place 为策略在线执行，同时更新两网络"]
    end
    D4 --> T1
    T4 --> INF["推理：M_pick 选 affordance 最高的拾取点 → M_place 在该拾取点条件下选最高放置点 → 执行 pick-and-place"]
```

图依据：论文 §4（4.1–4.7）/ Fig.2、Fig.3、Fig.5

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICCV 2023 |
| 方法名（清单标注） | DefoAfford |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2303.11057> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2303.11057)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2303.11057) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「DefoAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_learning-foresightful-dense-visual-affordance-for-deform.md`](../../sources/papers/affordance_awesome_learning-foresightful-dense-visual-affordance-for-deform.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2303.11057>
- [arXiv PDF](https://arxiv.org/pdf/2303.11057)：已通读方法部分（实验与附录以略读为准）：§4 方法（4.1–4.7）；§1–3 与实验部分未精读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2303.11057)
