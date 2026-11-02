---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordbot-3d-fine-grained-embodied-reasoning-via-multimo
arxiv: "2511.10017"
venue: "NeurIPS 2025"
summary: "NeurIPS 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_affordbot-3d-fine-grained-embodied-reasoning-via-multimo.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordBot

**AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models**（NeurIPS 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    IN["输入：3D 场景点云 S + 自然语言指令 T"]
    subgraph REP["Holistic Multimodal Representation"]
        V["Enriched Visual Synthesis<br/>场景中心 360° 水平环视，生成 N 个 surround-view 视图"]
        G["Geometry-Semantic Descriptors<br/>实例分割提取 affordance 元素<br/>位置 C、尺寸 Σ、语义类型 S"]
        A["3D-2D Associations<br/>3D 框投影到各视图并带 ID 标注<br/>adaptive labeling 避免标签重叠"]
        V --> A
        G --> A
    end
    subgraph COT["Chain-of-Thought（MLLM，observe-then-infer）"]
        S1["Step 1 Active View Selection<br/>选出能看清所有被指元素的视图"]
        S2["Step 2 Affordance Grounding<br/>输出目标元素 ID"]
        S3["Step 3 Motion Estimation<br/>输出运动类型与运动轴方向"]
        S1 --> S2 --> S3
    end
    IN --> V
    IN --> G
    A --> S1
    IN -.->|指令 T| S1
    S3 --> OUT["输出：(3D 实例 mask, 运动类型, 运动轴) 三元组"]
```

图依据：论文 §3 Methodology（§3.1 Overview、§3.2 Holistic Multimodal Representation、§3.3 Chain-of-thought Reasoning，Fig. 2 / Fig. 4）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | NeurIPS 2025 |
| 方法名（清单标注） | AffordBot |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2511.10017> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2511.10017)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2511.10017) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordBot」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordbot-3d-fine-grained-embodied-reasoning-via-multimo.md`](../../sources/papers/affordance_awesome_affordbot-3d-fine-grained-embodied-reasoning-via-multimo.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2511.10017>
- [arXiv PDF](https://arxiv.org/pdf/2511.10017)：已通读方法部分（实验与附录以略读为准）：§3 Methodology（§3.1–3.3）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2511.10017)
