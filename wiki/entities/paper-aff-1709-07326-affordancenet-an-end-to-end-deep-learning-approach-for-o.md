---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordancenet-an-end-to-end-deep-learning-approach-for-o
arxiv: "1709.07326"
venue: "ICRA 2018"
summary: "ICRA 2018 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_affordancenet-an-end-to-end-deep-learning-approach-for-o.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AffordanceNet

**AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection**（ICRA 2018）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual）**（清单未附方法摘要，此处不替原文概括贡献）。

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
flowchart LR
    IMG["输入 RGB 图像"] --> BB["VGG16 backbone 提取特征 (conv5_3)"]
    BB --> RPN["RPN (与 backbone 共享权重) 生成 RoI"]
    RPN --> RA["RoIAlign 池化为 7x7 特征"]
    BB --> RA
    RA --> FC["两层全连接 (各 4096)"]
    FC --> CLS["分类层: 物体类别"]
    FC --> REG["回归层: 物体框位置"]
    RA --> D1["conv + 反卷积 (7x7 到 30x30)"]
    D1 --> D2["conv + 反卷积 (到 122x122)"]
    D2 --> D3["conv + 反卷积 (到 244x244)"]
    D3 --> SM["逐像素 softmax: 多类 affordance mask"]
    CLS --> LOSS["多任务损失 L = Lcls + Lloc + Laff"]
    REG --> LOSS
    SM --> LOSS
    GT["真值 mask: 多阈值 robust resizing 到 244x244"] --> LOSS
```

图依据：论文 §III（III-B.1 RoIAlign、III-B.2 反卷积、III-B.3 robust resizing、III-B.4 端到端架构、III-C 多任务损失）/ Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICRA 2018 |
| 方法名（清单标注） | AffordanceNet |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文 | <https://arxiv.org/abs/1709.07326> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；模块与数据流见上文「流程总览」（依据论文方法部分），其余细节与实验请读 [原文](https://arxiv.org/abs/1709.07326)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/1709.07326) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AffordanceNet」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordancenet-an-end-to-end-deep-learning-approach-for-o.md`](../../sources/papers/affordance_awesome_affordancenet-an-end-to-end-deep-learning-approach-for-o.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/1709.07326>
- [arXiv PDF](https://arxiv.org/pdf/1709.07326)：已通读方法部分（实验与附录以略读为准）：Abstract、§III 方法、§III-D 训练与推理；实验 §IV 略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/1709.07326)
