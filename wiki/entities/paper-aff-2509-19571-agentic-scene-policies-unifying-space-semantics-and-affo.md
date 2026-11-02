---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-agentic-scene-policies-unifying-space-semantics-and-affo
arxiv: "2509.19571"
venue: "arXiv 2025"
summary: "arXiv 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_agentic-scene-policies-unifying-space-semantics-and-affo.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# ASP

**Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action**（arXiv 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action——在清单中归入 **Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    Q["语言指令 q_task"]
    X["RGB-D 观测 X_t"]
    M["F_VLM 构建开放词汇 Object Map M_t<br/>MobileSAM 分割 + CLIP 特征 + 深度反投影点云"]
    X --> M
    subgraph AG["LLM Agent π_ASP（LangChain + Gemini）：循环调用工具，上下文 c_t 累积工具输出；下列顺序为 Fig. 2 示例，实际顺序由 Agent 决定"]
        direction TB
        TG1["Grounding：τ_R 物体检索<br/>CLIP top-k + VLM 确认，写入 inventory I_t"]
        TG2["Grounding：空间推理工具<br/>distance_to / left / right / size_of 等"]
        SW["Skill Switch τ_π(o, q_a)"]
        UV["U_VLM 预测 affordance<br/>Step 1 选 skill 与部位描述<br/>Step 2 框 → SAM 2.1 → 3D 部位点云 P_part"]
        SK["执行 skill 工具<br/>grasp_part / pinch_pull / hook_pull / tip_push 等<br/>运动规划 + AnyGrasp"]
        TG1 --> TG2 --> SW --> UV --> SK
    end
    Q --> AG
    M --> AG
    SK -->|更新 I_t 与持物 h_t；反馈字符串让 Agent 重试| AG
    SK -->|后置条件失败：remapping 重建 M_t| M
    AG --> STOP["Agent 不再调用工具，episode 结束"]
```

图依据：论文 §III Preliminaries、§IV Method（Fig. 2）、§V Implementation Details（A Object Map、B Agent & Tools；移动版仅略读）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2025 |
| 方法名（清单标注） | ASP |
| 分组 1 | Reasoning · Reasoning Subcategory: Agentic Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2509.19571> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2509.19571)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2509.19571) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「ASP」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_agentic-scene-policies-unifying-space-semantics-and-affo.md`](../../sources/papers/affordance_awesome_agentic-scene-policies-unifying-space-semantics-and-affo.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2509.19571>
- [arXiv PDF](https://arxiv.org/pdf/2509.19571)：已通读方法部分（实验与附录以略读为准）：§III Preliminaries、§IV Method、§V-A/B 实现细节（§V-C Mobile ASP 略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2509.19571)
