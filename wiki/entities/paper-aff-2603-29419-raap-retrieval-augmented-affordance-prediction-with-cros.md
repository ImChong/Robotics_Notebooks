---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-raap-retrieval-augmented-affordance-prediction-with-cros
arxiv: "2603.29419"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_raap-retrieval-augmented-affordance-prediction-with-cros.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# RAAP

**RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
  IN[RGB-D 查询图 I_q + 任务标签 T]
  MEM[(Affordance Memory<br/>Grounded-SAM 裁剪目标 + CLIP 外观嵌入 f + 任务标签 + 2D affordance 标注<br/>来源 DROID / HOI4D)]
  IN --> R1[CLIP 文本编码器<br/>按任务相关性筛选记忆]
  MEM --> R1
  R1 --> R2[CLIP 图像余弦相似度<br/>检索 top-K 参考]
  R2 -->|仅 top-1| ST[静态 affordance：接触点<br/>Stable Diffusion 稠密特征匹配，取最大余弦相似位置 c_q]
  R2 -->|top-K| DY
  subgraph DY["动态 affordance：检索增强跨图动作对齐"]
    D1[共享 SigLIP-2 编码参考与查询图 patch token]
    D2[参考 token 经动作向量 MLP 产生 FiLM 调制<br/>+ 参考 ID 嵌入]
    D3[交叉注意力：查询 token 为 Q，K 个参考 token 为 K/V<br/>双重加权：门控网络 w_k x softmax 检索相似度]
    D4[拼接 CLS token → Transformer 编码器 → MLP 回归 2D 动作方向 a_q]
    D1 --> D2 --> D3 --> D4
  end
  ST --> LIFT
  DY --> LIFT[采样式 2D 到 3D 提升<br/>相机内参 + 点云得 3D 接触点与 3D 位移向量]
  LIFT --> GR[稠密抓取采样器，选最接近 3D 接触点的抓取位姿]
  GR --> IK[逆运动学 + 无碰撞轨迹]
  IK --> ACT[抓取后沿 3D 位移向量运动<br/>仿真位置控制 / 真机笛卡尔阻抗控制]
```

图依据：论文 §III-A 至 §III-D / Fig.2（Overview of RAAP）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | RAAP |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2603.29419> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.29419)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.29419) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「RAAP」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的是标题、出处、分类标签与「流程总览」所述方法流程；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_raap-retrieval-augmented-affordance-prediction-with-cros.md`](../../sources/papers/affordance_awesome_raap-retrieval-augmented-affordance-prediction-with-cros.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.29419>
- [arXiv PDF](https://arxiv.org/pdf/2603.29419)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§III Method（A 记忆、B 检索与静态迁移、C 动态对齐、D 执行）、§IV-A 实验设定。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.29419)
