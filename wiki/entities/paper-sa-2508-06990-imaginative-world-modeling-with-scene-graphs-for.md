---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2508.06990"
venue: "arXiv 2025"
summary: "Imaginative world modeling with scene graphs for embodied agent navigation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2508_06990_imaginative-world-modeling-with-scene-gr.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation

**Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 347/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Imaginative world modeling with scene graphs for embodied agent navigation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Imaginative world modeling with scene graphs for embodied agent navigation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["RGB-D 观测 ot"] --> B["符号世界建模：开放词表检测器识别物体 → 区域分组与区域 captioner → 楼层节点"]
    B --> C["分层场景图 gt = ϕ(gt-1, ot)"]
    C --> D["符号世界想象：占据图中未知区域中心 + 已观测节点投影到鸟瞰图 It"]
    D --> E["VLM 补全未观测区域语义，得到 ĝt = ψ(gt, It)"]
    E --> F["上下文感知目标验证器（VLM 查询含目标的图像）"]
    F -->|"确认目标"| G["3D 规划器（FMM）朝目标位置规划动作"]
    F -->|"未找到目标"| H["前沿检测器得到前沿集合"]
    H --> I["开发增益：前沿局部子图与目标类别的语义得分（CLIP 或 LLM）"]
    H --> J["探索增益：占据图上 raycasting 计算可见未知区域比例"]
    I --> K["回退机制：开发增益超过阈值 λ 则选之，否则选探索增益最大的前沿"]
    J --> K
    K --> G
    G --> L["执行动作，环境产生新观测"]
    L --> A
```

图依据：论文 §III（A 世界建模与想象、B 目标验证器、C 目标信息增益、D 3D 规划器）/ Fig.2、Fig.4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 347/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2508.06990> |

## 核心机制（归纳）

### 策展导读要点

Imaginative world modeling with scene graphs for embodied agent navigation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2508.06990) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **813 Navigation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2508_06990_imaginative-world-modeling-with-scene-gr.md`](../../sources/papers/sun_awesome_wm_2508_06990_imaginative-world-modeling-with-scene-gr.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2508.06990>
- [arXiv PDF](https://arxiv.org/pdf/2508.06990)：已通读方法部分（实验与附录以略读为准）：§III A–D（SGImagineNav 五个模块）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2508.06990)
