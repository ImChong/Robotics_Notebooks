---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordancegrasp-r1-leveraging-reasoning-based-affordance
arxiv: "2602.03547"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_affordancegrasp-r1-leveraging-reasoning-based-affordance.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordanceGrasp-R1

**AffordanceGrasp-R1: Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordanceGrasp-R1: Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」）。

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
    IN["输入：RGB 图像 I + 语言指令 T"] --> MLLM["MLLM 模块：Qwen2.5-VL-7B<br/>输出思考过程 + 结构化提示"]
    subgraph TRAIN["MLLM 两阶段后训练"]
        COT["CoT 冷启动数据（QVQ-72B 生成推理轨迹，Qwen2.5-VL-72B 蒸馏规则并校验）"] --> SFT["阶段1：SFT 冷启动"]
        SFT --> GRPO["阶段2：GRPO 强化学习<br/>奖励：格式 / 非重复 / 精度"]
    end
    GRPO -.-> MLLM
    MLLM --> BP["bounding box B + 关键点 P"]
    BP --> SAM["分割模块：LoRA 微调 SAM 2"]
    IN --> SAM
    SAM --> MASK["affordance mask M"]
    DEPTH["深度图 D"] --> PC["反投影得到全场景点云 S"]
    PC --> GAG["GA-Grasp 生成 6D 抓取候选"]
    MASK --> FILT["掩码引导的语义过滤<br/>+ 3D IoU 匹配"]
    GAG --> FILT
    FILT --> RANK["打分排序 + NMS"]
    RANK --> EXEC["最终可执行抓取位姿 → 真机执行"]
```

图依据：论文 §4（Method）/ Fig.4、Fig.5；CoT 数据生成见 §3 / Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | AffordanceGrasp-R1 |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2602.03547> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；机制见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2602.03547)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2602.03547) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordanceGrasp-R1」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordancegrasp-r1-leveraging-reasoning-based-affordance.md`](../../sources/papers/affordance_awesome_affordancegrasp-r1-leveraging-reasoning-based-affordance.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2602.03547>
- [arXiv PDF](https://arxiv.org/pdf/2602.03547)：已通读方法部分（实验与附录以略读为准）：§3 数据构建、§4.1 MLLM、§4.2 分割模块、§4.3 抓取模块

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2602.03547)
