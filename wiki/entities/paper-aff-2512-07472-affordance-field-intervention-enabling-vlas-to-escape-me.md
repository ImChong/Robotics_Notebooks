---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordance-field-intervention-enabling-vlas-to-escape-me
arxiv: "2512.07472"
venue: "CVPR 2026"
summary: "CVPR 2026 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordance-field-intervention-enabling-vlas-to-escape-me.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AFI

**Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation**（CVPR 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要；方法流程见「流程总览」）。

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
    subgraph SAF["空间 Affordance 场 SAF 构建"]
        TASK["任务指令 τ + RGB-D 观测"] --> GPT["GPT-4o 分解为有序子目标并提取目标词"]
        GPT --> GSAM["Grounded-SAM 得 2D mask<br/>结合深度反投影为目标点云"]
        GSAM --> VOX["体素化工作空间：目标引导场 + 避障场<br/>加权融合、距离变换 + 高斯平滑、归一化为代价场 V_SAF"]
    end
    VLA["VLA 策略正常执行"] --> DET{"记忆陷阱检测：末端近似静止<br/>且距目标仍远？"}
    DET -- 否 --> VLA
    DET -- 是 --> RB["历史回滚：回到历史缓冲中 SAF 代价最低的位置"]
    VOX -.-> DET
    VOX -.-> RB
    RB --> WP["阶段1：在回滚点邻域按 SAF 取代价最低的 N 个中间航点"]
    WP --> CAND["阶段2：逐航点查询 VLA，各生成 K 个动作块候选"]
    CAND --> SEL["正向运动学得末端轨迹，累计 SAF 代价最小者执行"]
```

图依据：论文 §3.2（SAF 构建）/ Fig.2、§4（Method）/ Fig.3；§4.1 记忆陷阱检测、§4.2.1 历史回滚、§4.2.2 分层探索。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2026 |
| 方法名（清单标注） | AFI |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2512.07472> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；机制见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2512.07472)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2512.07472) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AFI」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordance-field-intervention-enabling-vlas-to-escape-me.md`](../../sources/papers/affordance_awesome_affordance-field-intervention-enabling-vlas-to-escape-me.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2512.07472>
- [arXiv PDF](https://arxiv.org/pdf/2512.07472)：已通读方法部分（实验与附录以略读为准）：§3.2 Affordance 场构建、§4.1 记忆陷阱检测、§4.2 Affordance Field Intervention

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2512.07472)
