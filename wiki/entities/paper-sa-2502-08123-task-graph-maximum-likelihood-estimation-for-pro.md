---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2502.08123"
venue: "arXiv 2025"
summary: "Introduces a gradient-based approach for learning task graphs from procedural activities, optimizing edge weights via maximum likelihood"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2502_08123_task-graph-maximum-likelihood-estimation.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Task Graph Maximum Likelihood Estimation for Procedural Activity Understandin...

**Task Graph Maximum Likelihood Estimation for Procedural Activity Understanding in Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 133/249** 篇，分组 **292 Procedural Learning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces a gradient-based approach for learning task graphs from procedural activities, optimizing edge weights via maximum likelihood

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Introduces a gradient-based approach for learning task graphs from procedural activities, optimizing edge weights via maximum likelihood
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["关键步骤序列 y（自我中心视频的 key-step 序列）"] --> B["序列预处理：去重复，或映射为多条无重复序列"]
    T["关键步骤名称文本 / 视频片段"] --> U["EgoVLPv2 提取 D 维嵌入"]
    U --> V["加入可学习 S/E 嵌入，Transformer 编码器（DCEL 正则防表征塌缩）"]
    V --> W["Relation head：拼接所有节点对，关系 Transformer 逐步降维"]
    W --> X["评分矩阵 A，行 softmax 得到邻接矩阵 Z（DAG 掩码）"]
    B --> L["TGML 损失：最大化已观测步骤的边权，对比项压低未出现步骤到已观测步骤的边"]
    X --> L
    L -->|"梯度下降"| X
    X --> P["后处理：阈值 1/n 二值化，传递性约简，去环，补 START/END 边"]
    P --> Q["任务图 G（DAG）"]
```

图依据：论文 §III-A 任务图最大似然学习框架（Eq. 8-11）、§III-B / Fig. 3 Task Graph Transformer、§III-C 序列预处理、§III-D DAG 掩码、§III-E 输出图后处理。注意：页面 frontmatter 的 arXiv 号 2502.08123 对应的是另一篇无关论文（Provably Robust Federated Reinforcement Learning），本页标题所述论文的正确编号为 arXiv 2502.17753，本图按 2502.17753 绘制。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 133/249 |
| 分组 | 292 Procedural Learning |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.08123> |

## 核心机制（归纳）

### 策展导读要点

Introduces a gradient-based approach for learning task graphs from procedural activities, optimizing edge weights via maximum likelihood

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.08123) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **292 Procedural Learning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Task Graph Maximum Likelihood Estimation for Procedural Activity Understandin...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **292 Procedural Learning**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2502_08123_task-graph-maximum-likelihood-estimation.md`](../../sources/papers/sun_awesome_ego_2502_08123_task-graph-maximum-likelihood-estimation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2502.08123>
- [arXiv PDF（正确论文：2502.17753，与页面 arXiv 号 2502.08123 不符；2502.08123 是联邦强化学习鲁棒性论文）](https://arxiv.org/pdf/2502.17753)：已通读方法部分（实验与附录以略读为准）：§III-A 至 §III-E

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2502.08123)
