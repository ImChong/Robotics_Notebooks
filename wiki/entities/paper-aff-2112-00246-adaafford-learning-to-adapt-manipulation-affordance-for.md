---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-adaafford-learning-to-adapt-manipulation-affordance-for
arxiv: "2112.00246"
venue: "ECCV 2022"
summary: "ECCV 2022 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_adaafford-learning-to-adapt-manipulation-affordance-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AdaAfford

**AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-shot Interactions**（ECCV 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AdaAfford: Learning to Adapt Manipulation Affordance for 3D Articulated Objects via Few-shot Interactions——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    O["单帧部分点云 O<br/>关节体物体"] --> W["Where2Act 预训练网络<br/>无交互的 affordance 先验 A"]
    W --> AIP1["AIP 模块提出首个交互 u1 = (p, R)"]
    subgraph LOOP["测试时递归交互 t = 1, 2, ..."]
        EXE["仿真中执行 Where2Act 的任务轨迹<br/>观测部件运动 m，得到 I_t = (O, p, R, m)"]
        EXE --> SET["交互集合 I_t = I_(t-1) ∪ {I_t}"]
        SET --> AAP["AAP 模块<br/>Adaptive Information Encoder：逐交互编码 + 注意力加权平均 → 隐信息 z（128 维）"]
        AAP --> POST["Adaptive Affordance Network + Adaptive Critic Network<br/>输出 affordance 后验 A_I 与动作成功分数 s_AAP"]
        AAP --> AIP["AIP 模块<br/>AIP Affordance Network 选最高 actionability 的点 p<br/>在 p 采样 100 个动作，AIP Critic 选最高分的 R"]
        AIP --> NEXT["下一次交互动作 u_(t+1)"]
        NEXT --> EXE
    end
    AIP1 --> EXE
    POST --> STOP{"达到交互预算<br/>或最大 AIP actionability 低于阈值（如 0.05）"}
    STOP --> FIN["输出最终 affordance 后验 A_I = A_(I_T)"]
```

图依据：论文 §3 Problem Formulation、§4 Method / Fig.2、Fig.3（测试时流程见 §4 Test-time Overview，AAP 见 §4.1，AIP 与下一步动作选择见 §4.2，交替训练见 §4.3）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ECCV 2022 |
| 方法名（清单标注） | AdaAfford |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2112.00246> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2112.00246)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2112.00246) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AdaAfford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_adaafford-learning-to-adapt-manipulation-affordance-for.md`](../../sources/papers/affordance_awesome_adaafford-learning-to-adapt-manipulation-affordance-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2112.00246>
- [arXiv PDF](https://arxiv.org/pdf/2112.00246)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Problem Formulation、§4 Method（4.1–4.3）、§5 实验（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2112.00246)
