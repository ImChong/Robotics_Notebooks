---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordancesam-segment-anything-once-more-in-affordance-g
arxiv: "2504.15650"
venue: "arXiv 2025"
summary: "arXiv 2025 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_affordancesam-segment-anything-once-more-in-affordance-g.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordanceSAM

**AffordanceSAM: Segment Anything Once More in Affordance Grounding**（arXiv 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordanceSAM: Segment Anything Once More in Affordance Grounding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Generalizable）**（清单未附方法摘要；论文方法流程见下方「流程总览」）。

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
  I["输入：图像 + 文本提示 '动词 + 物体名'"] --> P1["路径 i：SAM 图像编码器 E_I → SAM 图像 token I_s"]
  I --> P2["路径 ii：BEIT-3 多模态编码器 E_M<br/>（图像 token + 文本 token + CLS）"]
  P2 --> CLS["CLS 输出 F_c → 提示编码器 E_P"]
  P2 -->|"文本特征"| Q["affordance queries 交叉注意力（含自注意力块）"]
  P1 -->|"多层全局注意力特征加权融合 F_v"| Q
  Q --> T["两层转置卷积 TransConv(Q_af)"]
  P1 --> D["SAM 掩码解码器 D"]
  CLS --> D
  T -->|"与 EVF-SAM 掩码特征相加"| D
  D --> M["输出：affordance map M_a"]
  S1["阶段 1：Part 1 二值掩码数据<br/>仅 EVF-SAM，微调多模态编码器与提示编码器"] --> S2["阶段 2：Part 2 伪标签 affordance map<br/>加入适配模块，仅冻结 SAM 图像编码器"]
  S2 --> S3["阶段 3：Part 3 人工标注 AGD20K<br/>损失与可训练模块同阶段 2"]
```

图依据：论文 §Approach（Preliminary / Affordance-Adaption Module / C2F-Aff 与三阶段训练）/ Fig.3、Table 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2025 |
| 方法名（清单标注） | AffordanceSAM |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Generalizable |
| 论文 | <https://arxiv.org/abs/2504.15650> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；方法流程见上方「流程总览」，更多机制与实验请读 [原文](https://arxiv.org/abs/2504.15650)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2504.15650) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordanceSAM」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_affordancesam-segment-anything-once-more-in-affordance-g.md`](../../sources/papers/affordance_awesome_affordancesam-segment-anything-once-more-in-affordance-g.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2504.15650>
- [arXiv PDF](https://arxiv.org/pdf/2504.15650)：已通读方法部分（实验与附录以略读为准）：§Approach（Preliminary、Affordance-Adaption Module、Coarse-to-Fine Dataset and Training 三阶段）、实验设定（Implementation Details）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2504.15650)
