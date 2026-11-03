---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "2010.03978"
venue: "Advances in data science and information engineering: proceedings from ICDATA"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Transition / Domain Adaptation。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_2010_03978_a-brief-review-of-domain-adaptation.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# A brief review of domain adaptation

**A brief review of domain adaptation** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 110/139** 篇，分组 **Transition / Domain Adaptation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Transition / Domain Adaptation。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Transition / Domain Adaptation。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

下图是论文自己的分类体系与章节结构（综述），节点沿用论文的分类名。

```mermaid
flowchart TD
    BG["§1 引言: 与迁移学习 / 半监督 / 多任务 / 多视图学习 / 领域泛化的关系<br/>§2 符号与定义: 域 D = X, Y, p x,y; 源域与目标域; 风险最小化"] --> CAT
    CAT["§3 领域适应的分类 Categorization of Domain Adaptation"] --> GAPC["按类别差距 category gap<br/>Closed set / Open set / Partial / Universal"]
    CAT --> SHIFT["闭集领域适应中的域偏移 domain shift<br/>Prior shift / Covariate shift / Concept shift"]
    GAPC --> APP["§4 Approaches (论文聚焦闭集无监督领域适应)"]
    SHIFT --> APP
    APP --> SHALLOW["Shallow 浅层架构: 实例 + 特征两类技术"]
    APP --> DEEP["4.3 Deep Domain Adaptation"]
    SHALLOW --> INST["4.1 Instance-Based Adaptation"]
    SHALLOW --> FEAT["4.2 Feature-Based Adaptation<br/>Subspace-based / Transformation-based / Reconstruction-based"]
    DEEP --> D1["4.3.1 Discrepancy-based"]
    DEEP --> D2["4.3.2 Reconstruction-based"]
    DEEP --> D3["4.3.3 Adversarial-based"]
```

图依据：综述分类体系图，论文 §3 Categorization of Domain Adaptation 的类别差距与域偏移分类、§4 Approaches 开头对浅层 / 深层方法的划分及 §4.1-4.3.3 小节标题。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 110/139 |
| 分组 | Transition / Domain Adaptation |
| 出处 | Advances in data science and information engineering: proceedings from ICDATA |
| 论文 | <https://arxiv.org/abs/2010.03978> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Transition / Domain Adaptation。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2010.03978) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Transition / Domain Adaptation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「A brief review of domain adaptation」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Transition / Domain Adaptation**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_2010_03978_a-brief-review-of-domain-adaptation.md`](../../sources/papers/lc_awesome_sim2real_2010_03978_a-brief-review-of-domain-adaptation.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/2010.03978>
- [arXiv PDF](https://arxiv.org/pdf/2010.03978)：已通读摘要、§1 引言、§2 符号与定义、§3 分类、§4 开头的方法总述，以及 §4.1 / §4.2 / §4.3 各节的开头定义段与 §4.3.3 对抗式方法；其余具体方法条目略读

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/2010.03978)
