---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-great-geometry-intention-collaborative-inference-for-ope
arxiv: "2411.19626"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image-language-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_great-geometry-intention-collaborative-inference-for-ope.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# GREAT

**GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image-language-3D）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    IMG["交互图像 I"] --> RES["ResNet → 图像特征 Fi"]
    PCL["物体点云 Pc"] --> PN["PointNet++ → 点特征 Fp"]
    IMG --> MLLM["微调的 InternVL（adapter，LoRA rank 16）<br/>Multi-Head Affordance CoT"]
    subgraph MH["Multi-Head Affordance Chain-of-Thought"]
        OH["Object-Head：<br/>1 交互部件感知 → 2 几何结构推理"]
        AH["Affordance-Head：<br/>3 交互详细描述 → 4 类比推理（再列两种交互）"]
    end
    MLLM --> OH
    MLLM --> AH
    OH --> RB1["Roberta 文本编码 → To"]
    AH --> RB2["Roberta 文本编码 → Ta"]
    RB1 --> XA["交叉注意力 + 自注意力<br/>对齐几何属性与交互意图"]
    RB2 --> XA
    XA --> TO["物体几何知识 To_bar"]
    XA --> TA["交互意图知识 Ta_bar"]
    TO --> CM["CMAFM：把几何知识注入 PointNet++ 最深层点特征<br/>→ Feature Propagation → Ftp"]
    PN --> CM
    TA --> FI["与图像特征融合 → Fti"]
    RES --> FI
    CM --> DEC["解码器：拼接 Ftp 与 Fti → 输出头 + sigmoid"]
    FI --> DEC
    DEC --> OUT["3D 物体 affordance 热图 φ<br/>损失：focal + dice"]
```

图依据：论文 §3.1 概述 / Fig.2、§3.2 Multi-Head Affordance CoT、§3.3 CMAFM、§3.4 解码器与损失

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | GREAT |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: image-language-3D |
| 论文 | <https://arxiv.org/abs/2411.19626> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（仅画论文方法部分明确写出的模块与数据流，不含推测），训练细节与实验请读 [原文](https://arxiv.org/abs/2411.19626)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2411.19626) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「GREAT」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_great-geometry-intention-collaborative-inference-for-ope.md`](../../sources/papers/affordance_awesome_great-geometry-intention-collaborative-inference-for-ope.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2411.19626>
- [arXiv PDF](https://arxiv.org/pdf/2411.19626)：已通读方法部分（实验与附录以略读为准）：§3（概述、Multi-Head Affordance Chain-of-Thought、Cross-Modal Adaptive Fusion Module、Decoder and Loss Functions）；§4 数据集 PIADv2 与实验略读，附录未读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2411.19626)
