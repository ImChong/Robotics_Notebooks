---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-dora-object-affordance-guided-reinforcement-learning-for
arxiv: "2505.14819"
venue: "IEEE CBS 2025"
summary: "IEEE CBS 2025 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_dora-object-affordance-guided-reinforcement-learning-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# DORA

**DORA: Object Affordance-Guided Reinforcement Learning for Dexterous Robotic Manipulation**（IEEE CBS 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

DORA: Object Affordance-Guided Reinforcement Learning for Dexterous Robotic Manipulation——在清单中归入 **Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart TB
  subgraph GEN["功能性抓取位姿生成（§III-B）"]
    AM["物体 affordance map"] --> VT["Voting-Based Grasp Classification<br/>逐接触点取最近邻 affordance 标签，<br/>逐指多数投票，再按手指权重加权投票"]
    GD["灵巧手抓取数据<br/>能量 E = EDFC + EHO + ERobot 优化<br/>附 contact semantic map"] --> VT
    VT --> CAND["带功能类别的抓取候选集 G"]
  end
  subgraph RLP["Affordance-Guided RL 管线（§III-C）"]
    CAND --> FIL["Motion-Feasibility Filtering<br/>运动规划检查可达且无碰撞"]
    FIL -->|"Gvalid 作为约束与先验"| SAC["SAC 策略（actor-critic）"]
    ST["状态：手/臂本体感知、物体位姿、<br/>affordance map、抓取候选、子任务阶段"] --> SAC
    SAC --> ACT["wrist 位姿增量（Jacobian 伪逆 IK 转关节速度）<br/>+ 手关节目标位置"]
    SUB["Sub-task Transition<br/>终止条件触发时切换子任务与奖励"] --> SAC
  end
  subgraph RW["奖励设计（§III-D）"]
    RF["R = wa·Raffordance + wt·Rtask − wp·Rpenalty"] --> SAC
  end
```

图依据：论文 §III-A–D / Fig.1、Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IEEE CBS 2025 |
| 方法名（清单标注） | DORA |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal |
| 论文 | <https://arxiv.org/abs/2505.14819> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」（依据论文方法部分，只画论文明确写出的模块与数据流）。完整机制与实验请读 [原文](https://arxiv.org/abs/2505.14819)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2505.14819) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「DORA」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_dora-object-affordance-guided-reinforcement-learning-for.md`](../../sources/papers/affordance_awesome_dora-object-affordance-guided-reinforcement-learning-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2505.14819>
- [arXiv PDF](https://arxiv.org/pdf/2505.14819)：已通读方法部分（实验与附录以略读为准）：§III 问题定义与方法（A–D）及实验任务设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2505.14819)
