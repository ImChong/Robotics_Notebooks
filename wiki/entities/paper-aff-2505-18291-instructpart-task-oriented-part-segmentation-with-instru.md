---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-instructpart-task-oriented-part-segmentation-with-instru
arxiv: "2505.18291"
venue: "ACL 2025"
summary: "ACL 2025 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image-language）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_instructpart-task-oriented-part-segmentation-with-instru.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# InstructPart

**InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning**（ACL 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image-language）**（清单未附方法摘要；数据集与评测流程见「流程总览」，贡献概括以原文为准）。

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
    SRC["图像来源: Flickr 与 AGD20K 中的免费许可图像<br/>按日常家务任务挑选, 物体类别分布均匀"] --> MASK["人工标注: 与任务相关的细粒度分割掩码 M<br/>以及物体名 O / 部件名 P / affordance / action"]
    SRC --> INS["6 位人类专家撰写自由形式任务指令<br/>指令中刻意不出现部件名"]
    INS --> GPT["GPT-4 改写以提升语法精确度与句式多样性<br/>再经人工核查, 防止幻觉"]
    MASK --> DS["InstructPart 数据集<br/>2,400 图 / 48 物体类 / 44 部件类 / 30 affordance / 37 action"]
    GPT --> DS
    DS --> TRPS["任务 TRPS: F 指令, 图像 → 掩码 M<br/>指令分人工标注 / GPT-4 改写两种"]
    DS --> ORPS["任务 ORPS: 直接给部件名<br/>格式一 P of O; 格式二 带 affordance 描述"]
    TRPS --> EVAL["评测现有 VLM: 开放词汇分割 / 指代分割 / 推理分割模型 / GPT-4V<br/>指标 gIoU, cIoU, P@50, P@50:95"]
    ORPS --> EVAL
    DS -->|"1,800 条样本微调"| PISA["基线 PISA: 在 LISA 上引入冻结 DINOv2 骨干<br/>线性层融合多层特征 → 源自 SAM 的图像解码器"]
    PISA --> EVAL
```

图依据：数据集构建与基准评测流程图，论文 §3.1 任务定义、§3.2 数据集、§3.3 基线方法、§4.1 指标、§4.2 被评模型、§4.4 微调、附录 A 数据来源。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ACL 2025 |
| 方法名（清单标注） | InstructPart |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: image-language |
| 论文 | <https://arxiv.org/abs/2505.18291> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；数据集构建、任务定义与基线评测流程见上文「流程总览」（依据论文 §3-§4，未逐页核对实验数字），其余细节与实验请读 [原文](https://arxiv.org/abs/2505.18291)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2505.18291) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「InstructPart」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述的数据集构建、任务定义与基线评测流程（依据论文 §3-§4，未逐页核对实验数字）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_instructpart-task-oriented-part-segmentation-with-instru.md`](../../sources/papers/affordance_awesome_instructpart-task-oriented-part-segmentation-with-instru.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2505.18291>
- [arXiv PDF](https://arxiv.org/pdf/2505.18291)：已通读 §1 引言、§3 任务定义 / 数据集 / 基线方法、§4.1-4.2 指标与被评模型、§4.4 微调、附录 A 数据集细节；§4.3 结果表、§5 讨论与附录 C-H 略读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2505.18291)
