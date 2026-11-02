---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-open-vocabulary-functional-3d-scene-graphs-for-real-worl
arxiv: "2503.19199"
venue: "CVPR 2025"
summary: "CVPR 2025 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: RGB-D/3D scene graph）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_open-vocabulary-functional-3d-scene-graphs-for-real-worl.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# FunGraph3D

**Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces**（CVPR 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Spatial） ｜ Datasets / Benchmarks（Data Modality: RGB-D/3D scene graph）**（清单未附方法摘要；论文方法流程见下方「流程总览」）。

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
  IN["输入：位姿 RGB-D 帧序列 {(I_i, D_i)}"] --> OBJ["RAM++ 识别物体标签 → GroundingDINO 检测物体框/mask/置信度"]
  OBJ --> LLM1["GPT-4 给出各物体的交互元件标签<br/>（过滤不可交互物体，如 wall、bed）"]
  LLM1 --> IE["GroundingDINO 用组合提示（如 door. handle）检测交互元件"]
  OBJ --> FUSE["利用深度与相机投影反投影并跨视角融合<br/>→ 3D 物体候选 C_o、交互元件候选 C_ie"]
  IE --> FUSE
  FUSE --> CAP["节点描述：选 Top-Nv 视图裁剪 → LLaVA v1.6 描述<br/>（交互元件：多尺度放大 + 红框高亮）→ GPT-4 汇总为语言描述 L"]
  CAP --> LOC["局部关系：空间重叠过滤 → LLM 判断刚性连接并生成关系描述"]
  LOC --> REM["远程关系：LLM 提议候选目标 → VLM 评估可行性 → LLM 给出置信度"]
  LOC --> G["合并局部与远程子图"]
  REM --> G
  G --> OUT["输出：功能性 3D 场景图 G = (O, I, R)"]
```

图依据：论文 §4.1–4.4 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2025 |
| 方法名（清单标注） | FunGraph3D |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Spatial |
| 分组 2 | Datasets / Benchmarks · Data Modality: RGB-D/3D scene graph |
| 论文 | <https://arxiv.org/abs/2503.19199> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」，更多机制与实验请读 [原文](https://arxiv.org/abs/2503.19199)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2503.19199) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「FunGraph3D」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_open-vocabulary-functional-3d-scene-graphs-for-real-worl.md`](../../sources/papers/affordance_awesome_open-vocabulary-functional-3d-scene-graphs-for-real-worl.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2503.19199>
- [arXiv PDF](https://arxiv.org/pdf/2503.19199)：已通读方法部分（实验与附录以略读为准）：§3 问题定义、§4 方法（4.1–4.4）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2503.19199)
