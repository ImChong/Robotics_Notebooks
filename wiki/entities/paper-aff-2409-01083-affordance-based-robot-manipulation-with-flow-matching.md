---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordance-based-robot-manipulation-with-flow-matching
arxiv: "2409.01083"
venue: "arXiv 2024"
summary: "arXiv 2024 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordance-based-robot-manipulation-with-flow-matching.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Affordance-based Robot Manipulation with Flow Matching

**Affordance-based Robot Manipulation with Flow Matching**（arXiv 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordance-based Robot Manipulation with Flow Matching——在清单中归入 **Action（Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  IMG["RGB-D 图像（同一场景）"]
  TXT["语言指令（如 comb the hair）"]
  PT["文本条件 prompt tuning：文本 Transformer 层的 prompt 前置到冻结的 ViT-B-16 视觉层（deep 结构逐层引入）"]
  DEC["轻量 Transformer decoder（单层）"]
  AFF["任务相关 affordance 热力图（2D）"]
  ENC["观测编码：ResNet 视觉嵌入 o（也可为状态输入或 RGB-D 图像）"]
  FM["Flow matching 策略：U-Net 流模型 v_θ(x_t, t|o)，FiLM 条件化"]
  SRC["随机 waypoints x0 ~ N(0, I)"]
  ACT["沿流从 t=0 积分到 t=1，得到目标动作轨迹（6D 末端轨迹、关节、夹爪）"]
  EXE["执行 T_a 步后重新规划（预测 T_p 步）"]
  IMG --> PT
  TXT --> PT
  PT --> DEC
  DEC --> AFF
  AFF --> ENC
  ENC --> FM
  SRC --> FM
  FM --> ACT
  ACT --> EXE
  EXE -.-> ENC
```

图依据：论文 Fig. 1、§III-A（Prompt Tuning）、§III-B（Flow Matching Policy）与 Fig. 3。affordance 图作为观测 o 的一种输入进入策略，依据 Fig. 1 / Fig. 3 图注；其余输入路径论文未明确，未画。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2024 |
| 方法名（清单标注） | - |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: explicit；Action Subcategory: Integrated policy · Explicit input |
| 论文 | <https://arxiv.org/abs/2409.01083> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），其余机制与实验请读 [原文](https://arxiv.org/abs/2409.01083)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2409.01083) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Affordance-based Robot Manipulation with Flow Matching」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_affordance-based-robot-manipulation-with-flow-matching.md`](../../sources/papers/affordance_awesome_affordance-based-robot-manipulation-with-flow-matching.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2409.01083>
- [arXiv PDF](https://arxiv.org/pdf/2409.01083)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III 方法（A、B）、§IV-A 数据集与基线设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2409.01083)
