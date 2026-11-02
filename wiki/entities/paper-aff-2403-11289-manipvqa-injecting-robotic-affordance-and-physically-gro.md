---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-manipvqa-injecting-robotic-affordance-and-physically-gro
arxiv: "2403.11289"
venue: "IROS 2024"
summary: "IROS 2024 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_manipvqa-injecting-robotic-affordance-and-physically-gro.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# ManipVQA

**ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Modal Large Language Models**（IROS 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

ManipVQA: Injecting Robotic Affordance and Physically Grounded Information into Multi-Modal Large Language Models——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（方法流程见下方「流程总览」）。

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
    subgraph DATA["指令数据集构建 (§III-B)"]
        D1["机器人 affordance：HANDAL、RGB-D Part Affordance"]
        D2["物理概念：PhysObjects（透明/储液/可密封）"]
        D3["通用视觉推理：PACO、RefCOCO、Visual Genome"]
        D4["GPT-4 生成复杂 affordance grounding 任务（Listing 1 提示）"]
    end
    D1 --> VQA["统一为 VQA 格式：REC / REG 及 REC-Grounding-Affordance、REC-Physical"]
    D2 --> VQA
    D3 --> VQA
    D4 --> VQA
    VQA --> FT["微调 SPHINX MLLM：LLaMA2 语言骨干<br/>CLIP 视觉编码器 + Q-Former，448 图切 4 个 224 子图<br/>视觉编码器冻结，统一交叉熵损失"]
    FT --> OUT["输出：目标区域 bounding box 或物理属性回答"]
    OUT --> POL["接机器人策略：SAM-HQ 由框得 mask<br/>mask 几何中心作为接触点 + 启发式策略 (SAPIEN 仿真)"]
```

图依据：论文 §III-B 数据集构建、§III-C 任务形式、§III-D 模型与微调、§IV-A-2 / Fig.6（接入机器人策略）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2024 |
| 方法名（清单标注） | ManipVQA |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2403.11289> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上方「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2403.11289)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2403.11289) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「ManipVQA」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_manipvqa-injecting-robotic-affordance-and-physically-gro.md`](../../sources/papers/affordance_awesome_manipvqa-injecting-robotic-affordance-and-physically-gro.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2403.11289>
- [arXiv PDF](https://arxiv.org/pdf/2403.11289)：已通读方法部分（实验与附录以略读为准）：§III 方法（affordance 与物理概念建模、数据集构建、任务形式、MLLM 微调策略）、§IV-A 实现细节、§IV-C 仿真中的机器人操作与 Fig.6 图注

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2403.11289)
