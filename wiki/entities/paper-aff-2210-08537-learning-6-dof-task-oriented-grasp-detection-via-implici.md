---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-6-dof-task-oriented-grasp-detection-via-implici
arxiv: "2210.08537"
venue: "IROS 2022"
summary: "IROS 2022 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_learning-6-dof-task-oriented-grasp-detection-via-implici.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visu...

**Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance**（IROS 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    CAM["RGB-D 单视角深度图 → 2.5D 局部点云 P"] --> IEN["隐式估计网络：多 stream PointNet++（每个 stream 对应一个 affordance 标签），隐式最大似然估计生成 SE(3) 抓取"]
    IEN --> GEN["多样抓取候选"]
    GEN --> GEV["抓取评估网络：PointNet++ 输入物体点云 + 夹爪点云，输出成功概率"]
    GEV --> GC["粗抓取候选 GC + 置信度 SO"]
    CAM --> VAP["视觉 affordance 预测模块：PointNet + DGCNN(EdgeConv) + MHSA 嵌入，MLP 度量解码器"]
    VAP --> HM["各任务的 3D affordance 热图 SM"]
    HM --> TOP["取热图值最高的 100 个点"]
    TOP --> SV["计算视觉引导分 SV：点到抓取中间控制点的最小 L2 距离"]
    GC --> SV
    GC --> FUSE["融合：SF = α1·SC + α2·SV（α 均取 0.5）"]
    SV --> FUSE
    FUSE --> GF["精抓取候选 GF → 变换到相机坐标系执行"]
```

图依据：论文 §III–§IV、§V-A（Fine Grasp Candidate Generation）/ Fig.1、Fig.3、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2022 |
| 方法名（清单标注） | - |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2210.08537> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2210.08537)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2210.08537) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visu...」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_learning-6-dof-task-oriented-grasp-detection-via-implici.md`](../../sources/papers/affordance_awesome_learning-6-dof-task-oriented-grasp-detection-via-implici.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2210.08537>
- [arXiv PDF](https://arxiv.org/pdf/2210.08537)：已通读方法部分（实验与附录以略读为准）：§III 问题定义、§IV 框架（数据集、抓取检测、视觉 affordance 模块、损失）、§V-A 实现细节与精抓取生成；§V 其余实验略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2210.08537)
