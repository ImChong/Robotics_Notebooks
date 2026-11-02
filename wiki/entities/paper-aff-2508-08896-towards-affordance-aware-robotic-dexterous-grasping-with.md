---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-towards-affordance-aware-robotic-dexterous-grasping-with
arxiv: "2508.08896"
venue: "AAAI 2026"
summary: "AAAI 2026 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_towards-affordance-aware-robotic-dexterous-grasping-with.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordDex

**Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors**（AAAI 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors——在清单中归入 **Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph S1["阶段 1：Human Hand Trajectory Imitating"]
        H["OakInk2 人手运动数据（约 2,200 条右手序列）"]
        PH["基策略 π^H<br/>奖励：指尖关键点模仿 + 动作平滑<br/>状态：机器人状态 + 物体状态 + 点云"]
        H --> PH
    end
    subgraph NAA["Negative Affordance-aware Segmentation（离线，每物体一次）"]
        T["过程式纹理化网格 → 六个方向渲染多视图图像"]
        G["GPT-4V 生成负 affordance 描述<br/>如 blade part"]
        SM["SAM 网格点提示 + NMS 得到候选 mask 集"]
        CL["模糊 mask 外区域 → CLIP 与文本相似度最高者为分割结果"]
        N["投影到 3D 点云得到负 affordance N_t"]
        T --> G --> CL
        T --> SM --> CL
        CL --> N
    end
    subgraph S2["阶段 2：Affordance-aware Residual Learning"]
        TP["状态式教师策略 π^T（PPO）<br/>输入含物体状态与 N_t，学习残差动作<br/>奖励含 r_n：惩罚靠近负 affordance"]
        ACT["最终动作 a = π^H + π^T（冻结 π^H）"]
        ST["视觉式学生策略 π^S<br/>输入：机器人状态 + 点云 + N_t<br/>DAgger 从教师蒸馏"]
        TP --> ACT
        TP --> ST
    end
    PH -->|冻结| ACT
    N --> TP
    N --> ST
    ST --> OUT["部署：视觉式灵巧抓取策略"]
```

图依据：论文 §3 Methodology（§3.1 Human Hand Trajectory Imitating、§3.2 Negative Affordance-aware Segmentation、§3.3 Affordance-aware Residual Learning，Fig. 2）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | AAAI 2026 |
| 方法名（清单标注） | AffordDex |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal |
| 论文 | <https://arxiv.org/abs/2508.08896> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2508.08896)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2508.08896) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordDex」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_towards-affordance-aware-robotic-dexterous-grasping-with.md`](../../sources/papers/affordance_awesome_towards-affordance-aware-robotic-dexterous-grasping-with.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2508.08896>
- [arXiv PDF](https://arxiv.org/pdf/2508.08896)：已通读方法部分（实验与附录以略读为准）：§3 Methodology（§3.1–3.3）、§4.1–4.3 数据集 / 指标 / 实现细节

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2508.08896)
