---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-diffusion-models-are-open-world-affordance-learners-leve
arxiv: "2508.01651"
venue: "ECCV 2026"
summary: "ECCV 2026 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_diffusion-models-are-open-world-affordance-learners-leve.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# DAG

**Diffusion Models are Open-World Affordance Learners: Leveraging Generative Priors for 3D Affordance Learning**（ECCV 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Diffusion Models are Open-World Affordance Learners: Leveraging Generative Priors for 3D Affordance Learning——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart LR
  I["HOI 参考图像"] --> ASE["Affordance Semantics Encoder<br/>冻结 CLIP 图像编码器 + 多层加权 + MLP"]
  I -->|"加噪 x_t，单次前向"| UNet["冻结文生图扩散 U-Net"]
  ASE -->|"隐式语义 Ak 作为条件"| UNet
  UNet -->|"多尺度特征 Av"| Agg["聚合网络<br/>按层加权混合"]
  Agg -->|"affordance priors Ag"| AB
  T["affordance 文本"] --> TE["冻结 CLIP 文本编码器"]
  TE -->|"At"| AB["Affordance Blocks<br/>缩放 + 自注意力 + 交叉注意力 + FFN + 池化"]
  AB -->|"affordance embeddings Af"| Dec
  P["点云"] --> PE["预训练点编码器<br/>+ 几何引导上采样/特征传播"]
  PE -->|"CLS token"| Dec["Multi-Source Affordance Decoder<br/>CLS 与 Af 交叉注意力，再与点特征交叉注意力，MLP"]
  PE -->|"点特征 fp"| Dec
  Dec --> M["3D affordance mask"]
```

图依据：论文 §3.1–3.6 / Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ECCV 2026 |
| 方法名（清单标注） | DAG |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Generalizable |
| 论文 | <https://arxiv.org/abs/2508.01651> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（依据论文方法部分，只画论文明确写出的模块与数据流）。完整机制与实验请读 [原文](https://arxiv.org/abs/2508.01651)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2508.01651) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「DAG」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_diffusion-models-are-open-world-affordance-learners-leve.md`](../../sources/papers/affordance_awesome_diffusion-models-are-open-world-affordance-learners-leve.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2508.01651>
- [arXiv PDF](https://arxiv.org/pdf/2508.01651)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1–3.6）与实验设定（§4.1–4.2）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2508.01651)
