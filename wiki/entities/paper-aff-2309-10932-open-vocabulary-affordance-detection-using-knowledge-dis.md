---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-open-vocabulary-affordance-detection-using-knowledge-dis
arxiv: "2309.10932"
venue: "ICRA 2024"
summary: "ICRA 2024 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_open-vocabulary-affordance-detection-using-knowledge-dis.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Po...

**Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlation**（ICRA 2024）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlation——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）**（方法流程见下方「流程总览」）。

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
    PC["输入：点云 C (n 个点)"] --> TE["教师网络（大数据集预训练，冻结）：逐点嵌入"]
    PC --> ST["学生网络（轻量）：逐点嵌入 P1..Pn"]
    TE --> GEO["FPS + KNN 构造局部区域，得点间相对关系 R"]
    ST --> GEO
    GEO --> L1["几何迁移损失 Lgeo_transfer：教师与学生的 R 做 MSE"]
    TE --> CAT["Cross-Attention Projector → 教师注意力图 Ωte"]
    ST --> CAS["Cross-Attention Projector → 学生注意力图 Ωst"]
    CAT --> L2["注意力迁移损失 Latt_transfer = MSE(Ωte, Ωst)"]
    CAS --> L2
    LAB["开放词表 affordance 文本标签（如 take a seat, sittable, drivable）"] --> TXT["CLIP 文本编码器（冻结）：文本嵌入 T"]
    ST --> COR["Text-Point Correlation：点与文本相关性 w(i,j)<br/>加权求和得文本注意力特征 T̂i"]
    TXT --> COR
    COR --> SM["逐点 softmax 得各 affordance 概率<br/>加权负对数似然损失"]
    SM --> OUT["输出：逐点 affordance 预测"]
```

图依据：论文 §III-A~C / Fig.2（两个分支：point-point attention 知识蒸馏 + text-point correlation）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICRA 2024 |
| 方法名（清单标注） | - |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Generalizable |
| 论文 | <https://arxiv.org/abs/2309.10932> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上方「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2309.10932)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2309.10932) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Po...」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_open-vocabulary-affordance-detection-using-knowledge-dis.md`](../../sources/papers/affordance_awesome_open-vocabulary-affordance-detection-using-knowledge-dis.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2309.10932>
- [arXiv PDF](https://arxiv.org/pdf/2309.10932)：已通读方法部分（实验与附录以略读为准）：§III 方法全部（A 概述、B 点-点注意力知识蒸馏、C 文本-点相关性）、§IV-A 实验设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2309.10932)
