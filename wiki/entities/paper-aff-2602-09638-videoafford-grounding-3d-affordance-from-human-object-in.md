---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-videoafford-grounding-3d-affordance-from-human-object-in
arxiv: "2602.09638"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: video-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_videoafford-grounding-3d-affordance-from-human-object-in.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# VideoAfford

**VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: video-3D）**（清单未附方法摘要，此处不替原文概括贡献）。

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
  V[HOI 视频 V：采样 N 帧] --> VE[Video-LLaVA 视频编码器 e<br/>稀疏编码得视频 token]
  V --> AE[预训练潜在动作编码器 m<br/>每帧压缩为 2 个动作 token A_c]
  T[文本指令 T] --> LLM
  VE --> LLM[Video-LLaVA 的 LLM（视频 MLLM g）<br/>视频 token 与动作 token 拼接作为输入]
  AE --> LLM
  LLM --> J[输出理解文本 J<br/>含特殊 token AFF（承载 affordance 世界知识）]
  J --> Q[取 AFF 的隐状态 H_aff<br/>投影为查询嵌入 A_m]
  PC[物体点云 P_c] --> PE[预训练 3D 点编码器 f → P_s<br/>几何引导上采样（Point-BERT 式）→ 稠密点特征 P_d]
  subgraph DEC["Transformer 轻量 affordance 解码器"]
    CA[交叉注意力：A_m 为 Query，P_d 为 Key/Value → A_f]
    ML[MLP → 逐点 affordance mask A_mask]
    CA --> ML
  end
  Q --> CA
  PE --> CA
  ML --> OUT[3D affordance 分割结果]
  subgraph LOSS["训练目标（端到端）"]
    L[文本交叉熵 L_ce + BCE + IoU + 空间连续性 Dice 损失 L_spatial]
  end
  J -.-> L
  ML -.-> L
```

图依据：论文 §4.1–§4.3 / Fig.4（VideoAfford 架构）；点编码器与动作编码器的具体预训练模型见论文补充材料，未读。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | VideoAfford |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: video-3D |
| 论文 | <https://arxiv.org/abs/2602.09638> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2602.09638)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2602.09638) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「VideoAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_videoafford-grounding-3d-affordance-from-human-object-in.md`](../../sources/papers/affordance_awesome_videoafford-grounding-3d-affordance-from-human-object-in.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2602.09638>
- [arXiv PDF](https://arxiv.org/pdf/2602.09638)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§4 Methods（4.1 概览、4.2 网络结构、4.3 训练目标）；§5 实验与补充材料未读/略读。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2602.09638)
