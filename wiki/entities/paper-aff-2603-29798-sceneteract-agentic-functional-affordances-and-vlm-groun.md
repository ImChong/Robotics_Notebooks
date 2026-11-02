---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-sceneteract-agentic-functional-affordances-and-vlm-groun
arxiv: "2603.29798"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_sceneteract-agentic-functional-affordances-and-vlm-groun.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# SceneTeract

**SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes——在清单中归入 **Reasoning（Reasoning Subcategory: Agentic Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  T[活动描述 T<br/>自然语言高层意图] --> C
  A[智能体画像 A<br/>w_clear / h_eye / h_arm / r_arm / 行走或轮式] --> C
  S[3D 场景 S = 物体集 O + 几何 G] --> C
  C[多模态输入上下文 C<br/>文本序列化 Π_txt + 俯视渲染 Π_vis] --> VLM
  subgraph SD["阶段一：语义分解"]
    VLM[VLM Φ<br/>在封闭原子动作库 A 中规划] --> PLAN[语义计划 π<br/>动作-物体步骤序列 a_t, o_t]
  end
  subgraph GV["阶段二：几何验证 Ψ"]
    FAM[按动作族取固定有序属性清单<br/>Mobility / Contact / Handling / Perception]
    CHK[逐项布尔几何检查<br/>navigable / reachable / interactable / clearance / visible]
    FAM --> CHK
  end
  PLAN --> FAM
  CHK --> STEP[步骤结论 v_t = 各检查合取]
  STEP --> LABEL[活动标签 = 各步骤合取]
  CHK --> REP[诊断报告 R<br/>失败步骤与量化原因]
  LABEL --> REP
  REP --> U1[应用：审计 3D 场景]
  REP --> U2[应用：评测 VLM 可行性判断]
  REP --> U3[应用：验证器奖励 + GRPO 后训练 VLM]
```

图依据：论文 §3.1–§3.3 / Fig.1、Fig.3、Tab.1–2（各动作族内部检查顺序见 Tab.1，图中从简）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | SceneTeract |
| 分组 1 | Reasoning · Reasoning Subcategory: Agentic Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2603.29798> |

## 核心机制（归纳）

清单本身没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验与细节请读 [原文](https://arxiv.org/abs/2603.29798)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.29798) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「SceneTeract」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_sceneteract-agentic-functional-affordances-and-vlm-groun.md`](../../sources/papers/affordance_awesome_sceneteract-agentic-functional-affordances-and-vlm-groun.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.29798>
- [arXiv PDF](https://arxiv.org/pdf/2603.29798)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Method（3.1 输入三元组、3.2 验证接口、3.3 执行流程、验证器反馈后训练）。

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.29798)
