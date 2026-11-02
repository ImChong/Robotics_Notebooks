---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-2d-invariant-affordance-knowledge-for-3d-afford
arxiv: "2408.13024"
venue: "AAAI 2025"
summary: "AAAI 2025 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Interaction-driven）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_learning-2d-invariant-affordance-knowledge-for-3d-afford.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# MIFAG

**Learning 2D Invariant Affordance Knowledge for 3D Affordance Grounding**（AAAI 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning 2D Invariant Affordance Knowledge for 3D Affordance Grounding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Interaction-driven）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  IMGS["n 张参考图像 I_1..I_n（同一 affordance 类别、外观各异）"]
  PC["3D 点云 P_c，点云特征 P_in"]
  subgraph IAM["IAM：不变 affordance 知识提取（多层迭代）"]
    Q["可学习的 affordance query tokens：对图像特征做多头 cross-attention 更新"]
    MLP["MLP 对齐聚合各图像的 query，得到 Q_f"]
    FB["图像特征分支：以 Q_f 做 cross-attention + 多头 self-attention 更新"]
    SIM["逐层对各图像特征施加相似度损失 L_Sim"]
  end
  DICT["最后一层 query 分支输出：Invariant Affordance Knowledge Dictionary"]
  subgraph ADM["ADM：字典自适应融合"]
    IQ["IQDCA：点云特征为 query，字典为 key/value，余弦相似度加权得 P_q"]
    SW["self-weighted attention 丢弃无关 token 得 P_mix"]
  end
  FUSE["P_mix 与 P_in 结合得 P_out"]
  DEC["Affordance decoder f_d → 点级 affordance 预测 P_pred"]
  CLS["IAM 末层图像特征池化 → affordance 类别 logits ŷ（L_CE）"]
  IMGS --> Q
  Q --> MLP
  MLP --> FB
  FB --> Q
  FB --> SIM
  Q --> DICT
  DICT --> IQ
  PC --> IQ
  IQ --> SW
  SW --> FUSE
  PC --> FUSE
  FUSE --> DEC
  FB --> CLS
```

图依据：论文 §3.1–3.4（Overview、IAM、ADM、Decoder and Loss）与 Fig. 2；P_in 的具体点云编码器未在所读段落中明确，图中不展开。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | AAAI 2025 |
| 方法名（清单标注） | MIFAG |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Interaction-driven |
| 论文 | <https://arxiv.org/abs/2408.13024> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），其余机制与实验请读 [原文](https://arxiv.org/abs/2408.13024)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2408.13024) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「MIFAG」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_learning-2d-invariant-affordance-knowledge-for-3d-afford.md`](../../sources/papers/affordance_awesome_learning-2d-invariant-affordance-knowledge-for-3d-afford.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2408.13024>
- [arXiv PDF](https://arxiv.org/pdf/2408.13024)：已通读方法部分（实验与附录以略读为准）：§3.1–3.4 方法

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2408.13024)
