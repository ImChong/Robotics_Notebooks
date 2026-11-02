---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-ego-topo-environment-affordances-from-egocentric-video
arxiv: "2001.04583"
venue: "CVPR 2020"
summary: "CVPR 2020 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Interaction-driven）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_ego-topo-environment-affordances-from-egocentric-video.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Ego-Topo

**Ego-Topo: Environment Affordances from Egocentric Video**（CVPR 2020）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Ego-Topo: Environment Affordances from Egocentric Video——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Interaction-driven）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
  V["第一人称长视频\n(EPIC-Kitchens / EGTEA+)"] --> L["3.1 区域定位网络 L\nSiamese ResNet-18 + 5 层 MLP\n预测两帧是否属于同一 zone"]
  L --> G["3.2 构建拓扑 affordance 图 (Algorithm 1)\n逐帧与已有节点比较相似度\n合并进节点 / 新建节点, 并连边"]
  G --> LK["跨视频、跨环境链接\n按节点动作/物体分布的 KL 相似度\n层次聚类得到合并后的图"]
  G --> AF
  LK --> AF["3.3 环境 affordance 学习\n节点标签 = 各次访问中交互的并集\n多标签分类 (2 层 MLP + 线性 + sigmoid, BCE)"]
  G --> AN
  LK --> AN["3.4 长时程动作预测\n节点特征经 GCN 融合邻居信息\n平均后线性分类预测未来动作"]
```

图依据：论文 §3（Ego-Topo Approach，§3.1–3.4，Fig.1 / Fig.4、Algorithm 1）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2020 |
| 方法名（清单标注） | Ego-Topo |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Interaction-driven |
| 论文 | <https://arxiv.org/abs/2001.04583> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；论文自述的方法流程见上方「流程总览」，实验数字仍以原文为准；需要机制与实验请读 [原文](https://arxiv.org/abs/2001.04583)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2001.04583) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Ego-Topo」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_ego-topo-environment-affordances-from-egocentric-video.md`](../../sources/papers/affordance_awesome_ego-topo-environment-affordances-from-egocentric-video.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2001.04583>
- [arXiv PDF](https://arxiv.org/pdf/2001.04583)：已通读方法部分（实验与附录以略读为准）：§3.1 区域发现、§3.2 拓扑图构建、§3.3 环境 affordance 推断、§3.4 长视频动作预测、§4 数据集与基线设定

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2001.04583)
