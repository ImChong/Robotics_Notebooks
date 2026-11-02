---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-a4a-cross-embodiment-transfer-of-action-oriented-4d-affo
arxiv: "2609.05892"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_a4a-cross-embodiment-transfer-of-action-oriented-4d-affo.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A4A

**A4A: Cross-Embodiment Transfer of Action-Oriented 4D Affordances from Human Demonstrations**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A4A: Cross-Embodiment Transfer of Action-Oriented 4D Affordances from Human Demonstrations——在清单中归入 **Action（Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation）**（清单未附方法摘要；方法流程见下方「流程总览」）。

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
    H["人类演示数据<br/>现有 HOI 数据 + 自采 RGB-D，共 80K+ 交互片段"] --> DC["4D 亲和数据构建<br/>GroundingDINO 定位 → SAM 2 掩码 → CoTracker3 跟踪查询点<br/>→ 深度回投影得 3D 点轨迹"]
    DC --> S["样本 S = 图像序列 + 指令 l + 交互相关查询点 3D 轨迹"]
    subgraph P1["阶段 I：交互中心 4D 亲和预训练"]
        IN1["语言 + 图像 + 查询点坐标"] --> E1["原生 VLM 编码 + 点投影器 P_Q"]
        E1 --> F1["共享隐变量模块 F_theta"]
        F1 --> HQ["4D 轨迹头 H_Q：预测未来 4D 点运动（残差于匀速预测）"]
    end
    S --> IN1
    subgraph P2["阶段 II：机器人动作微调"]
        IN2["语言 + 图像 + 机器人状态"] --> E2["同一 VLM 编码 + 状态投影器 P_S"]
        E2 --> F2["共享隐变量模块 F_theta（由阶段 I 初始化）"]
        F2 --> HA["机器人动作头 H_A：动作块"]
    end
    F1 -.沿用 VLM 编码栈与 F_theta 参数.-> F2
```

图依据：论文 §3.2、§3.3 与 Fig. 2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | A4A |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: implicit；Action Subcategory: Integrated policy · Implicit representation |
| 论文 | <https://arxiv.org/abs/2609.05892> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法模块与数据流见上方「流程总览」（依据论文方法部分）。训练细节与实验请读 [原文](https://arxiv.org/abs/2609.05892)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2609.05892) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A4A」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 清单可确证的只有标题、出处与分类标签；方法流程见「流程总览」，实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_a4a-cross-embodiment-transfer-of-action-oriented-4d-affo.md`](../../sources/papers/affordance_awesome_a4a-cross-embodiment-transfer-of-action-oriented-4d-affo.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2609.05892>
- [arXiv PDF](https://arxiv.org/pdf/2609.05892)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.3）、§4.1 实验设置

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2609.05892)
