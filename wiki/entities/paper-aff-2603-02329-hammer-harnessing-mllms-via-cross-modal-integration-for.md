---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-hammer-harnessing-mllms-via-cross-modal-integration-for
arxiv: "2603.02329"
venue: "CVPR 2026"
summary: "CVPR 2026 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Interaction-driven） ｜ Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_hammer-harnessing-mllms-via-cross-modal-integration-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# HAMMER

**HAMMER: Harnessing MLLMs via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding**（CVPR 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

HAMMER: Harnessing MLLMs via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Interaction-driven） ｜ Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    IMG["交互图像 I + 文本提示 T（含物体类别）"] --> MLLM["MLLM F_θ（Qwen2.5-VL，LoRA 微调）"]
    MLLM --> HID["最后一层隐状态 h"]
    MLLM --> TXT["文本响应：affordance 标签（辅助 L_txt 监督）"]
    HID --> CONT["[CONT] token 隐状态 → MLP ψ_c → 意图嵌入 f_c"]
    PC["点云 P"] --> ENC["点云编码器（PointNet++）→ 瓶颈特征"]
    HID --> PROJ["MLP ϕ_h 投影 → f_h"]
    ENC --> CA["瓶颈阶段：Cross-Attn（query=点特征，key/value=f_h）"]
    PROJ --> CA
    CA --> DECP["点云解码器：多尺度特征 + 全分辨率 f_p"]
    PROJ --> GATE["门控加权汇聚得全局描述子 f_hg"]
    DECP --> FUS["特征层：拼接 f_p 与复制的 f_hg，经 MLP → 增强点特征"]
    GATE --> FUS
    CONT --> LIFT["多粒度几何提升：逐尺度 Cross-Attn（q=f_c，k/v=多尺度点特征）+ FFN"]
    DECP -->|"多尺度点特征"| LIFT
    LIFT --> F3D["3D 感知意图嵌入 f_c3D"]
    FUS --> DEC["Affordance 解码器：point-to-intention attention + MLP + sigmoid"]
    F3D --> DEC
    DEC --> OUT["点级 affordance 图 p（L_aff = focal + dice）"]
```

图依据：论文 §3.1–3.5（Fig. 2，Eq. 1–12）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2026 |
| 方法名（清单标注） | HAMMER |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Interaction-driven |
| 分组 2 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 论文 | <https://arxiv.org/abs/2603.02329> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2603.02329)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2603.02329) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「HAMMER」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_hammer-harnessing-mllms-via-cross-modal-integration-for.md`](../../sources/papers/affordance_awesome_hammer-harnessing-mllms-via-cross-modal-integration-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2603.02329>
- 论文 PDF：<https://arxiv.org/pdf/2603.02329>（已通读方法部分（实验与附录以略读为准）：方法 §3、实验设定 §4.1）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2603.02329)
