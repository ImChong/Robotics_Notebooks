---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-compassad-intent-driven-3d-affordance-grounding-in-funct
arxiv: "2604.02060"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language-3D）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_compassad-intent-driven-3d-affordance-grounding-in-funct.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# CompassAD

**CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: language-3D）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart LR
  P[多物体点云 P] --> E1[Uni3D 点编码器<br/>逐点特征 F]
  Q[隐含意图自然语言 Query] --> E2[RoBERTa 文本编码器<br/>token 特征 T]
  subgraph ICI["ICI 实例边界交叉注入"]
    G1[1 实例分组<br/>半径图连通分量得实例标签]
    G2[每实例内 FPS 选 Ng 中心<br/>取 M 近邻成 region，均值池化 + MLP]
    G3[2 区域-语言交叉注意力<br/>text 末尾追加可学习背景 token]
    G4[3 门控区域到点传播<br/>近邻 kp=3 反距离加权，乘性门调制点特征]
    G1 --> G2 --> G3 --> G4
  end
  P --> G1
  E1 --> G2
  E2 --> G3
  G4 --> D[Transformer 解码器]
  D --> O[逐点 affordance 概率 ŷ<br/>负例查询可输出空 mask]
  subgraph BCR["BCR 双层对比精炼（仅训练）"]
    B1[TG-Softmax<br/>区域级：排序与意图最匹配的区域]
    B2[TP-HardNeg<br/>点级：压制混淆表面上的高分误激活]
  end
  G3 -.-> B1
  G4 -.-> B2
  E2 -.-> B1
  E2 -.-> B2
```

图依据：论文 §4（Method）/ §4.1 ICI、§4.2 BCR / Fig.3（Overall architecture of CompassNet）；BCR 的具体连线按 §4.2 文字推断。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | CompassAD |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: language-3D |
| 论文 | <https://arxiv.org/abs/2604.02060> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2604.02060)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2604.02060) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「CompassAD」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_compassad-intent-driven-3d-affordance-grounding-in-funct.md`](../../sources/papers/affordance_awesome_compassad-intent-driven-3d-affordance-grounding-in-funct.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2604.02060>
- [arXiv PDF](https://arxiv.org/pdf/2604.02060)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 任务与基准构建、§4 Method（4.1 ICI、4.2 BCR）、§5.3 消融与 §5.4 机器人应用（略读）。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2604.02060)
