---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-what-can-i-do-here-learning-new-skills-by-imagining-visu
arxiv: "2106.00671"
venue: "ICRA 2021"
summary: "ICRA 2021 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_what-can-i-do-here-learning-new-skills-by-imagining-visu.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# VAL

**What Can I Do Here? Learning New Skills by Imagining Visual Affordances**（ICRA 2021）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

What Can I Do Here? Learning New Skills by Imagining Visual Affordances——在清单中归入 **Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  D["先验数据 D\n来自先前环境的成功轨迹"] --> A["A. 离线 affordance 学习\nVQVAE 学潜空间 z = φ(s)\n条件 PixelCNN 学 p(zt | z0)"]
  D --> B["B. 离线行为学习\nAWAC 训练目标条件策略 π(a | z, zg)\n稀疏目标到达奖励 + 目标重标记"]
  A --> B
  N["新环境 Enew\n观测初始图像 s0, z0 = φ(s0)"] --> C1["C. 在线行为学习\n从 affordance 模型采样目标 zg ~ pθ(· | z0)"]
  A --> C1
  C1 --> C2["执行 π(a | z, zg)\n收集探索数据入 replay buffer"]
  B --> C2
  C2 --> C3["off-policy RL 微调 π 与 Q"]
  C3 --> C1
```

图依据：论文 §V（Visuomotor Affordance Learning，A/B/C 三阶段）、Algorithm 1 / Fig.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICRA 2021 |
| 方法名（清单标注） | VAL |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal |
| 论文 | <https://arxiv.org/abs/2106.00671> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；论文自述的方法流程见上方「流程总览」，训练细节仍以原文为准；需要机制与实验请读 [原文](https://arxiv.org/abs/2106.00671)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2106.00671) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「VAL」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_what-can-i-do-here-learning-new-skills-by-imagining-visu.md`](../../sources/papers/affordance_awesome_what-can-i-do-here-learning-new-skills-by-imagining-visu.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2106.00671>
- [arXiv PDF](https://arxiv.org/pdf/2106.00671)：已通读方法部分（实验与附录以略读为准）：§III–V（问题设定与 VAL 的三个学习阶段）、§VI–VII 实验设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2106.00671)
