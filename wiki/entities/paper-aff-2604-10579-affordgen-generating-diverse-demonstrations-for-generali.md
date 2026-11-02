---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordgen-generating-diverse-demonstrations-for-generali
arxiv: "2604.10579"
venue: "CVPR 2026"
summary: "CVPR 2026 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordgen-generating-diverse-demonstrations-for-generali.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordGen

**AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Affordance Correspondence**（CVPR 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Affordance Correspondence——在清单中归入 **Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    SD["单条专家示范"] --> PRE["源示范预处理：抓取时刻 t_grasp、技能段 τ_s<br/>affording point 与 function point 关键点<br/>SAM2 分割点云，去背景并 FPS 下采样"]
    MESH["大规模 3D 网格（数据集或 3D 生成模型）"] --> NORM["6D 位姿估计归一化到规范空间<br/>多视角并行渲染"]
    PRE --> COR["DINOv2 特征余弦相似度做 3D 网格语义对应<br/>加权平均得目标关键点 x'"]
    NORM --> COR
    COR --> RP["关键点约束的轨迹重放：<br/>抓取段 τ_g 相对 affording point，技能段相对 function point 迁移"]
    RP --> TP["过渡段：运动规划或球面线性插值"]
    RP --> PC["Point cloud digital cousin：仿真渲染机器人与物体点云<br/>替换源示范中的对应部分，IK 求关节位置"]
    TP --> DATA["大规模多样化演示数据集"]
    PC --> DATA
    DATA --> POL["训练闭环视觉运动策略"]
```

图依据：论文 §3.1 Problem Formulation / §3.2 Source Demonstration Pre-processing / §3.3 Semantic Correspondence on 3D Meshes / §3.4 Diverse Demonstration Generation / Fig.1、3–5（策略训练仅见摘要表述）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2026 |
| 方法名（清单标注） | AffordGen |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation |
| 论文 | <https://arxiv.org/abs/2604.10579> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2604.10579)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2604.10579) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordGen」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordgen-generating-diverse-demonstrations-for-generali.md`](../../sources/papers/affordance_awesome_affordgen-generating-diverse-demonstrations-for-generali.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2604.10579>
- [arXiv PDF](https://arxiv.org/pdf/2604.10579)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.4、§4.1 实验设置

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2604.10579)
