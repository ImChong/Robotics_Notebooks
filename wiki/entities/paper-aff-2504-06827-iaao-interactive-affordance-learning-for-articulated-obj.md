---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-iaao-interactive-affordance-learning-for-articulated-obj
arxiv: "2504.06827"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_iaao-interactive-affordance-learning-for-articulated-obj.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# IAAO

**IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial）**（清单未附方法摘要；论文方法流程见下方「流程总览」）。

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
  I["输入：两个状态 t、t' 的多视角位姿图像"] --> S1
  subgraph S1["阶段 1：语义场景重建（每个状态独立）"]
    A1["SfM 稀疏点云初始化 3DGS"] --> A2["SAM 类无关实例/部件 mask"]
    A2 --> A3["mask 图聚类 → 3D 一致标签场 L_label"]
    A2 --> A4["MaskCLIP + DINOv2 特征 → 低维层级特征场（MLP 解码器，L_feat）"]
  end
  S1 --> S2
  subgraph S2["阶段 2：affordance 与运动预测"]
    B1["语言查询（CLIP 文本编码）：物体级 → 部件级查询 3D Gaussians"] --> B2["functional affordance 部件 mask"]
    B1 --> B3["静态 Gaussians + GeoTransformer 粗配准 → 全局变换 ξ_g"]
    B3 --> B4["各铰接部件局部变换 ξ_o：L_rgb + L_mask + L_match 联合优化"]
  end
  S2 --> S3["阶段 3：场景状态融合（按估计变换合并两个状态的 3DGS，可选微调）"]
  S3 --> O["输出：带 affordance 与关节运动参数的可交互重建场景"]
```

图依据：论文 §3 Our Method（3.1–3.3）/ Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | IAAO |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial |
| 论文 | <https://arxiv.org/abs/2504.06827> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」，更多机制与实验请读 [原文](https://arxiv.org/abs/2504.06827)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2504.06827) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「IAAO」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_iaao-interactive-affordance-learning-for-articulated-obj.md`](../../sources/papers/affordance_awesome_iaao-interactive-affordance-learning-for-articulated-obj.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2504.06827>
- [arXiv PDF](https://arxiv.org/pdf/2504.06827)：已通读方法部分（实验与附录以略读为准）：§3 Our Method（概述、3.1 语义场景重建、3.2 affordance 与运动预测、3.3 场景状态融合）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2504.06827)
