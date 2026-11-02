---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-afford-x-generalizable-and-slim-affordance-reasoning-for
arxiv: "2503.03556"
venue: "arXiv 2025"
summary: "arXiv 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_afford-x-generalizable-and-slim-affordance-reasoning-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Afford-X

**Afford-X: Generalizable and Slim Affordance Reasoning for Task-oriented Manipulation**（arXiv 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Afford-X: Generalizable and Slim Affordance Reasoning for Task-oriented Manipulation——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    IMG["图像 Xv"] --> VE["预训练视觉编码器"]
    TXT["任务描述 Xl"] --> TE["文本编码器"]
    VE --> FV["视觉特征 Fv"]
    TE --> FT["文本特征 Ft"]
    FT --> VA["VA 模块<br/>强调任务动作的文本特征"]
    VA --> FTP["增强文本特征 Ft'"]
    FTP --> BF["BF 模块<br/>视觉-文本双向融合"]
    FV --> BF
    BF --> ENC["Transformer 编码器<br/>全局自注意力"]
    ENC --> DEC["Transformer 解码器<br/>可学习 object queries"]
    DEC --> HEAD["并行输出头"]
    HEAD --> BOX["边界框 Bpred"]
    HEAD --> MASK["分割掩码 Mpred"]
    HEAD --> PREF["偏好分数 Spred<br/>由 no-object logit 得到"]
    subgraph DIST["训练：名词-代词蒸馏"]
        T["教师模型<br/>输入含物体名词的描述"] --> MB["记忆库 + K-means 聚类中心"]
        MB --> CS["学生模型 cluster selector<br/>最近邻原型替换代词特征"]
        T --> KL["soft binary target loss<br/>KL 对齐教师与学生 logits"]
        KL --> S["学生模型<br/>输入无类别描述"]
        CS --> S
    end
    S --> INF["推理：仅用学生模型 + 固定记忆库"]
```

图依据：论文 §III-B / Fig.4（整体结构）、§III-C / Fig.5（名词-代词蒸馏）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2025 |
| 方法名（清单标注） | Afford-X |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2503.03556> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（仅画论文方法部分明确写出的模块与数据流，不含推测），训练细节与实验请读 [原文](https://arxiv.org/abs/2503.03556)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2503.03556) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Afford-X」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_afford-x-generalizable-and-slim-affordance-reasoning-for.md`](../../sources/papers/affordance_awesome_afford-x-generalizable-and-slim-affordance-reasoning-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2503.03556>
- [arXiv PDF](https://arxiv.org/pdf/2503.03556)：已通读方法部分（实验与附录以略读为准）：§III（III-A 问题定义、III-B 模型结构、III-C 名词-代词蒸馏、III-E 小结）；VA 模块细节仅略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2503.03556)
