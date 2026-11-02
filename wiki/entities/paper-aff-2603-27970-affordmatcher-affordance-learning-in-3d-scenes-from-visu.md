---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordmatcher-affordance-learning-in-3d-scenes-from-visu
arxiv: "2603.27970"
venue: "CVPR 2026"
summary: "CVPR 2026 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: image-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_affordmatcher-affordance-learning-in-3d-scenes-from-visu.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordMatcher

**AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers**（CVPR 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: image-3D）**（清单未附方法摘要，此处不替原文概括贡献）。

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
  IMG[RGB 图像 I<br/>含 n 个交互线索 visual signifiers b_i：人体部位 / 物体 / 交互特征] --> FI[视觉特征 F_I]
  SC[3D 场景点云<br/>体素化 64^3，预训练 3D 模型分割出二值 mask 候选区域] --> FP[候选区域特征 F_P<br/>含动作描述 a_j]
  subgraph IM["4.1 实例匹配与 3D 推理"]
    CA[双向跨模态注意力<br/>W_M：signifier 引导的空间关键点特征<br/>W_R：来自 3D 上下文的推理反馈]
  end
  FI --> CA
  FP --> CA
  CA --> DIS["4.2 相异度矩阵 D<br/>W_M 与 W_R 的余弦相异度"]
  DIS --> SOFT[软阈值：D_ij 小于 0.2 的高相似对可多次传播<br/>处理一对多对应]
  DIS --> FF[展平并投影为 X<br/>FastFormer 式加性自注意力 → Match2Match 注意力图]
  SOFT -.-> FF
  FF --> HEAD[边界框与 mask 预测头]
  HEAD --> OUT[场景中的 affordance 实例 mask]
  subgraph LOSS["4.3 跨模态 affordance 学习（训练损失）"]
    L1[嵌入单位超球归一化 + 正则 L_embed]
    L2[对齐损失 L_align<br/>匹配输出对 S-CLIP 伪目标]
  end
  FF -.-> L2
```

图依据：论文 §4 / §4.1–§4.3、§5.1 实现细节（Fig.1 仅为任务示意；§4.3 中 Eq.9–10 两项损失图中未列）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2026 |
| 方法名（清单标注） | AffordMatcher |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Spatial |
| 分组 2 | Datasets / Benchmarks · Data Modality: image-3D |
| 论文 | <https://arxiv.org/abs/2603.27970> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.27970)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.27970) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordMatcher」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordmatcher-affordance-learning-in-3d-scenes-from-visu.md`](../../sources/papers/affordance_awesome_affordmatcher-affordance-learning-in-3d-scenes-from-visu.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.27970>
- [arXiv PDF](https://arxiv.org/pdf/2603.27970)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§4 AffordMatcher（4.1 实例匹配与 3D 推理、4.2 相异度量化、4.3 跨模态学习中 Eq.7–8）、§5.1 实验设定；§3 数据集构建（AffordBridge）仅略读。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.27970)
