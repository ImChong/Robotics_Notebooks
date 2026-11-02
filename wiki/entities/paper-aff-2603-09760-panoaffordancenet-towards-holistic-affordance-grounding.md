---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-panoaffordancenet-towards-holistic-affordance-grounding
arxiv: "2603.09760"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Visual） ｜ Datasets / Benchmarks（Data Modality: panoramic image/text）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_panoaffordancenet-towards-holistic-affordance-grounding.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# PanoAffordanceNet

**PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Visual） ｜ Datasets / Benchmarks（Data Modality: panoramic image/text）**（清单未附方法摘要，此处不替原文概括贡献）。

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
  IMG[360° ERP 全景图] --> VE[DINOv2 ViT-B/14 视觉编码器<br/>注意力层插入 LoRA → patch 特征 F_v]
  TXT[C 个 affordance 类别] --> TE[CLIP 文本编码器 + CoOp 提示学习器<br/>→ 文本嵌入 F_t]
  subgraph DASM["DASM 畸变感知频谱调制器"]
    D1[多头交叉注意力：F_v 为 Query，F_t 为 Key/Value<br/>语义注入 → F_v']
    D2[双频分解：Laplacian 高频 F_h / 高斯平滑低频 F_l]
    D3[HFEM 高频增强：锐化赤道区交互边界，抑制极区伪影]
    D4[LFSM 低频稳定：保持极区全局结构一致]
    D5[混合门控融合：语言驱动通道门 g_ch x 自适应空间门 g_sp<br/>门控残差相加 → F_freq]
    D6[上下文重聚合：MHSA + FFN → F_v'']
    D1 --> D2
    D2 --> D3
    D2 --> D4
    D3 --> D5
    D4 --> D5
    D5 --> D6
  end
  VE --> D1
  TE --> D1
  subgraph DEC["空间感知层级解码器"]
    G1[全局语义发现：轻量 Transformer 解码器<br/>F_t 为 Query 交叉注意 F_v'' → 初始图 A_init]
    G2[OSDH 全向球面致密化头<br/>F_v'' 球面投影得余弦亲和矩阵 S，top-k 种子 + 置信度去噪 + 最大传播]
    G1 --> G2
  end
  D6 --> G1
  D6 --> G2
  TE --> G1
  G2 --> OUT[致密连续的 affordance 热图 A_refined]
  subgraph LOSS["多层级训练目标"]
    L[L_BCE 像素级 + L_KL 分布拓扑一致 + L_RTC 区域-文本 InfoNCE 对比]
  end
  OUT -.-> L
```

图依据：论文 §III-A 至 §III-E / Fig.2（Overview of PanoAffordanceNet）、Fig.3（OSDH）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | PanoAffordanceNet |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Visual |
| 分组 2 | Datasets / Benchmarks · Data Modality: panoramic image/text |
| 论文 | <https://arxiv.org/abs/2603.09760> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.09760)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.09760) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「PanoAffordanceNet」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_panoaffordancenet-towards-holistic-affordance-grounding.md`](../../sources/papers/affordance_awesome_panoaffordancenet-towards-holistic-affordance-grounding.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.09760>
- [arXiv PDF](https://arxiv.org/pdf/2603.09760)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III Method（A 概览至 E 训练目标）、§V 实现细节开头；§IV 数据集 360-AGD 与实验结果仅略读。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.09760)
