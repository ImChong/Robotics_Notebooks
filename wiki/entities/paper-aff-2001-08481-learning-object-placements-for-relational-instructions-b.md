---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-object-placements-for-relational-instructions-b
arxiv: "2001.08481"
venue: "ICRA 2020"
summary: "ICRA 2020 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Relation-Based Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_learning-object-placements-for-relational-instructions-b.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Learning Object Placements For Relational Instructions by Hallucinating Scene...

**Learning Object Placements For Relational Instructions by Hallucinating Scene Representations**（ICRA 2020）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning Object Placements For Relational Instructions by Hallucinating Scene Representations——在清单中归入 **Reasoning（Reasoning Subcategory: Relation-Based Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  subgraph TR["训练阶段"]
    L["标注数据: RGB 图像 + 物体框注意力掩码\n+ 空间关系标签 (inside / left / right / in front / behind / on top)"] --> RN["阶段 1: 训练辅助 RelNet\nResNet-18, 交叉熵分类两物体空间关系"]
    X["RGB 图像 + 参照物体注意力掩码"] --> SR["Spatial-RelNet 编码-解码网络\n输出逐像素关系概率图 Γ"]
    SR --> SMP["按 Γ 采样位置 (u, v)"]
    SMP --> IMP["在 RelNet 的 conv3_x 特征图上\n于 (u, v) 植入 subject 物体高层特征\n幻觉出场景表示"]
    RN --> IMP
    IMP --> CLS["RelNet 分类幻觉场景\n得到关系后验 fϕ"]
    CLS --> LOSS["MSE 损失: Γ(u,v) 与 fϕ 对齐\n用于训练 Spatial-RelNet"]
    LOSS --> SR
  end
  subgraph INF["测试阶段"]
    I2["RGB 图像 + 参照物体注意力掩码"] --> SR2["Spatial-RelNet (RelNet 弃用)"]
    SR2 --> PM["逐像素放置分布 → 选取放置位置"]
  end
```

图依据：论文 §III-A/B/C/D（RelNet、Hallucinating Scene Representations、Learning pixelwise item placement distributions、Implementation Details，Fig.2 / Fig.3）；末节「选取放置位置」为对逐像素分布用途的概括

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICRA 2020 |
| 方法名（清单标注） | - |
| 分组 1 | Reasoning · Reasoning Subcategory: Relation-Based Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2001.08481> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；论文自述的方法流程见上方「流程总览」，实验数字仍以原文为准；需要机制与实验请读 [原文](https://arxiv.org/abs/2001.08481)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2001.08481) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Learning Object Placements For Relational Instructions by Hallucinating Scene...」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_learning-object-placements-for-relational-instructions-b.md`](../../sources/papers/affordance_awesome_learning-object-placements-for-relational-instructions-b.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2001.08481>
- [arXiv PDF](https://arxiv.org/pdf/2001.08481)：已通读方法部分（实验与附录以略读为准）：§III 方法（辅助 RelNet、特征植入幻觉、Spatial-RelNet 与实现细节）、§IV-A 数据集与评测设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2001.08481)
