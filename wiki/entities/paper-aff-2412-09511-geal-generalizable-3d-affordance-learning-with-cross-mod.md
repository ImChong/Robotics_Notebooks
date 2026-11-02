---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-geal-generalizable-3d-affordance-learning-with-cross-mod
arxiv: "2412.09511"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_geal-generalizable-3d-affordance-learning-with-cross-mod.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# GEAL

**GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    PC["物体点云 P"] --> GS["3D Gaussian Splatting<br/>高斯均值 = 点坐标，协方差/不透明度固定"]
    GS --> IMG["V 个视角的渲染图像 I<br/>另可按 affordance 分数渲染 2D mask"]
    PC --> B3["3D 骨干 PointNet++<br/>多尺度点特征"]
    IMG --> B2["2D 骨干 DINOv2<br/>多尺度图像特征"]
    Q["指令 Q"] --> L3["轻量语言模型 → h3D"]
    Q --> L2["视角相关 prompt Q2D<br/>→ 轻量语言模型 → h2D"]
    B2 --> GAFM2["GAFM 2D<br/>灵活粒度聚合（带噪声门控 softmax 权重）<br/>+ 文本条件视觉对齐"]
    L2 --> GAFM2
    B3 --> GAFM3["GAFM 3D<br/>逐尺度文本条件对齐 → 上采样 → 粒度聚合"]
    L3 --> GAFM3
    GAFM2 --> CAM["2D-3D Consistency Alignment Module<br/>映射到共享嵌入空间，MSE 一致性损失 Lconsis"]
    GAFM3 --> CAM
    GAFM2 --> D2["Transformer 解码器（3 层）<br/>文本为 query，视觉为 key/value"]
    GAFM3 --> D3["共享结构的解码器"]
    D2 --> Y2["2D 预测 ŷ2D = sigmoid(h_enh · f_enh)"]
    D3 --> Y3["3D affordance 分数 y ∈ R^N"]
    Y3 --> INF["推理时仅使用 3D 分支"]
```

图依据：论文 §3 概述 / Fig.2、§3.1 3D-2D 高斯映射、§3.2 跨模态一致性对齐 / Fig.3、§3.3 解码与训练（两阶段）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | GEAL |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Generalizable |
| 论文 | <https://arxiv.org/abs/2412.09511> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（仅画论文方法部分明确写出的模块与数据流，不含推测），训练细节与实验请读 [原文](https://arxiv.org/abs/2412.09511)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2412.09511) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「GEAL」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_geal-generalizable-3d-affordance-learning-with-cross-mod.md`](../../sources/papers/affordance_awesome_geal-generalizable-3d-affordance-learning-with-cross-mod.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2412.09511>
- [arXiv PDF](https://arxiv.org/pdf/2412.09511)：已通读方法部分（实验与附录以略读为准）：§3（概述、3D-2D Mapping with Gaussian Splatting、Cross-Modal Consistency Alignment、Decoding 与 Training）；§4 实验略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2412.09511)
