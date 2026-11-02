---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordance-benchmark-for-mllms
arxiv: "2506.00893"
venue: "ICASSP 2026"
summary: "ICASSP 2026 的 affordance 论文，清单分组：Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/affordance_awesome_affordance-benchmark-for-mllms.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# A4Bench

**Affordance Benchmark for MLLMs**（ICASSP 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordance Benchmark for MLLMs——在清单中归入 **Reasoning（Reasoning Subcategory: Language-Centric Affordance Reasoning） ｜ Datasets / Benchmarks（Data Modality: image）**（清单未附方法摘要；基准构建与评测协议见「流程总览」，贡献概括以原文为准）。

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
    PRIN["设计原则: 同时考察 constitutive 与 transformative affordance"] --> CON["Constitutive affordance: 1,282 题<br/>4 大类 9 个子学科: Daily Life / Natural and Rural / Urban and Industrial / Medical and Scientific"]
    PRIN --> TRA["Transformative affordance: 718 题<br/>Misleading / Time-dependent / Cross-cultural / Individual-variant"]
    ANN["人工专家标注: 60 人 = 10 资深专家 + 20 初级研究员 + 30 普通研究员<br/>资深专家出 transformative 题, 其余出基础题<br/>每题至少由 5 位其他标注者审核"] --> BENCH
    CON --> BENCH["A4Bench: 2,000 道多模态问答<br/>Yes-or-No 占 7.8%, What 占 92.2%"]
    TRA --> BENCH
    DIFF["难度保障三策略<br/>人类 - MLLM 混合迭代修订至模型答错 / 把物体名替换为 this object / 不公开正确选项个数"] --> BENCH
    BENCH --> PROT["评测协议: 零样本, 输入含概念提示 + 格式提示 + 问题 + 选项<br/>选项与正确答案随机化, 以 prompt-human progressive choice 校验回答"]
    PROT --> MLLM["17 个 MLLM: 9 个闭源 + 8 个开源"]
    PROT --> HUM["人类被试: 受控实验室, 记录最好 / 最差表现"]
    MLLM --> METRIC["指标: exact match accuracy, 另设随机猜测基线"]
    HUM --> METRIC
```

图依据：基准构建与评测协议图，论文 Fig. 2（A4Bench 结构）、Constructing the A4Bench（Key Principles / Focused Aspects / Question Collection / Human Expert Annotation / Question Response）、Experiment Results（Benchmark Candidates / Human Performance）、Table 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICASSP 2026 |
| 方法名（清单标注） | A4Bench |
| 分组 1 | Reasoning · Reasoning Subcategory: Language-Centric Affordance Reasoning |
| 分组 2 | Datasets / Benchmarks · Data Modality: image |
| 论文 | <https://arxiv.org/abs/2506.00893> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要；基准构建与评测协议见上文「流程总览」（依据论文 Constructing the A4Bench 与 Experiment Results 的设定部分，未核对实验数字），其余细节与实验请读 [原文](https://arxiv.org/abs/2506.00893)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2506.00893) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「A4Bench」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述的基准构建与评测协议（依据论文 Constructing the A4Bench 与 Experiment Results 的设定部分，未核对实验数字）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_affordance-benchmark-for-mllms.md`](../../sources/papers/affordance_awesome_affordance-benchmark-for-mllms.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2506.00893>
- [arXiv PDF](https://arxiv.org/pdf/2506.00893)：已通读 Introduction、Constructing the A4Bench（Key Principles 至 Question Response）、Experiment Results 的 Benchmark Candidates 与 Human Performance 设定；各模型结果与 Findings 及其后章节未读

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2506.00893)
