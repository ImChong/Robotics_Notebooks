---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordancevla-a-vision-language-action-model-empowering
arxiv: "2606.06155"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordancevla-a-vision-language-action-model-empowering.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordanceVLA

**AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding——在清单中归入 **Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    I["输入：观测 O_t + 语言指令 l"] --> U["Understanding Expert M_und<br/>得到 instruction-aware 表征 h_und"]
    U --> G["Affordance Generation Expert M_gen<br/>预测结构化 affordance Â_t"]
    G --> W1["Which2Act：目标框裁剪的视觉 latent<br/>MSE 重建"]
    G --> W2["Where2Act：2D affordance 热图<br/>BCE"]
    G --> W3["How2Act：3D 形状 latent 扩散去噪<br/>+ 10-DoF layout 回归"]
    W1 --> A
    W2 --> A
    W3 --> A
    U --> A["Action Expert M_act<br/>生成动作块 â_t:t+k"]
    S["本体状态 s_t"] --> A
    subgraph 训练["三阶段训练"]
        T1["Stage I：通用 affordance grounding 预训练<br/>仅训练 Affordance Generation Expert"] --> T2["Stage II：机器人数据协同训练<br/>L_act + L_afd 端到端"] --> T3["Stage III：目标任务后训练<br/>affordance 权重退火"]
    end
```

图依据：论文 §3.1 AffordanceVLA Architecture / §3.2 Affordance Knowledge Prediction / §3.3 Training Strategy / Fig.2（UAA 注意力：Affordance 专家只读 Understanding 专家，Action 专家读前两者）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | AffordanceVLA |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation |
| 论文 | <https://arxiv.org/abs/2606.06155> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2606.06155)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2606.06155) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordanceVLA」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_affordancevla-a-vision-language-action-model-empowering.md`](../../sources/papers/affordance_awesome_affordancevla-a-vision-language-action-model-empowering.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2606.06155>
- [arXiv PDF](https://arxiv.org/pdf/2606.06155)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.3（架构、affordance 预测、三阶段训练）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2606.06155)
