---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-grounding-by-remembering-cross-scene-and-in-scene-memory
arxiv: "2605.11616"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_grounding-by-remembering-cross-scene-and-in-scene-memory.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordMEM

**Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Grounding by Remembering: Cross-Scene and In-Scene Memory for 3D Functional Affordances——在清单中归入 **Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    IN["功能查询 q + 带位姿 RGB-D 序列"] --> A["(A) 冻结 VLM 解析查询<br/>三元组：参考物体 ℓ_ctx、交互部件 ℓ_int、空间限定 σ"]
    A -->|ℓ_ctx 与 ℓ_int 构造文本 prompt| S1["SAM3 初始候选 mask<br/>限定搜索空间"]
    subgraph B["(B) Cross-scene affordance memory"]
        MB["记忆库（按类别）：已见场景 3D 标注投影到帧<br/>深度一致性过滤，质量打分取 top-K，叠加成 masked image"]
        MB -->|按 ℓ_int 字符串匹配召回| VP["VLM 看示例图 + 查询帧<br/>输出正框（可操作核心）与负框"]
    end
    A --> VP
    S1 --> VP
    VP -->|仅正框送回 SAM3| M2["细粒度 2D 分割 mask"]
    M2 --> POOL["反投影融合为 3D 候选池<br/>可见度/前景支持阈值 + DBSCAN 合并"]
    subgraph C["(C) In-scene spatial memory"]
        G["场景图：节点含 3D 质心、AABB、类型 CTX 或 INT<br/>INT 节点链接到父 CTX 节点"]
        G --> LLM["LLM 结合 JSON 图、俯视渲染、节点 RGB 裁剪<br/>按空间限定 σ 选节点"]
    end
    POOL --> G
    A -->|σ 传给实例选择| LLM
    LLM --> OUT["输出：匹配实例的 3D mask"]
```

图依据：论文 §3 Method（§3.1 Functional Query Decomposition、§3.2 Cross-Scene Affordance Memory、§3.3 In-Scene Spatial Memory）/ Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | AffordMEM |
| 分组 1 | Reasoning · Reasoning Subcategory: Agentic Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2605.11616> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2605.11616)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2605.11616) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordMEM」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_grounding-by-remembering-cross-scene-and-in-scene-memory.md`](../../sources/papers/affordance_awesome_grounding-by-remembering-cross-scene-and-in-scene-memory.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2605.11616>
- [arXiv PDF](https://arxiv.org/pdf/2605.11616)：已通读方法部分（实验与附录以略读为准）：§3（含 3.1–3.3）、Fig.2

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2605.11616)
