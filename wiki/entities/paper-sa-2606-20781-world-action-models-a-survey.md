---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.20781"
venue: "arXiv 2026"
summary: "First systematic survey of World Action Models (WAMs), clarifying boundaries between world models, video generation models, action-grounded video world models, VLA policies, and WAMs."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_20781_world-action-models-a-survey.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# World Action Models

**World Action Models: A Survey** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 542/571** 篇，分组 **Surveys & Overviews**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

First systematic survey of World Action Models (WAMs), clarifying boundaries between world models, video generation models, action-grounded video world models, VLA policies, and WAMs.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- First systematic survey of World Action Models (WAMs), clarifying boundaries between world models, video generation models, action-grounded video world models, VLA policies, and WAMs.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S["综述：World Action Models（WAM）<br/>先厘清 VLA / 世界模型 / 视频生成 / WAM 的边界（§2）"] --> V1["视角一：三种设计哲学（§3）<br/>按「各方法必须生成什么」"]
    S --> V2["视角二：四轴解剖（§4）<br/>每个 WAM 记为 4 元组"]
    V1 --> P1["Render-and-Decode<br/>生成到像素再解码动作"]
    V1 --> P2["Latent-Only<br/>像素解码前停止，用中间表示"]
    V1 --> P3["Video-Generation-Free<br/>不经视频生成"]
    V2 --> A1["预测基底 Predictive Substrate<br/>像素 / 特征 / 几何基元 / affordance 图"]
    V2 --> A2["动作耦合 Action Coupling<br/>动作条件 rollout / 联合生成 / 预测后动作头"]
    V2 --> A3["架构骨干 Architectural Backbone<br/>迭代去噪 / 自回归 / 联合嵌入预测 / 混合 / LLM-VLM"]
    V2 --> A4["部署形态 Deployment Regime<br/>开环 rollout / 分块闭环 / 单步闭环 / 交互式模拟器"]
    V1 --> C["核心性质（§5）<br/>交互性 / 因果性 / 持久性 / 物理合理性 / 泛化"]
    V2 --> C
    C --> D["数据与评测（§6）→ 开放挑战（§7）"]
```

图依据：综述分类体系图（论文自身的组织结构），论文 §1 引言的章节路线、§3 三种设计哲学（Fig. 2）、§4 四轴解剖、§5–§7 目录。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 542/571 |
| 分组 | Surveys & Overviews |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.20781> |

## 核心机制（归纳）

### 策展导读要点

First systematic survey of World Action Models (WAMs), clarifying boundaries between world models, video generation models, action-grounded video world models, VLA policies, and WAMs.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.20781) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Surveys & Overviews** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World Action Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Surveys & Overviews**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- [视频世界模型综述](./paper-video-world-model-survey.md) — 从视频理解、生成与交互视角补充世界模型分类。

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2606_20781_world-action-models-a-survey.md`](../../sources/papers/sun_awesome_wm_2606_20781_world-action-models-a-survey.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.20781>
- [arXiv PDF](https://arxiv.org/pdf/2606.20781)：已通读目录、§1 引言、§2 World Action Models 的由来、§3 三种设计哲学开头与 §4 各小节标题（其余章节与表格以略读为准）；流程图依据论文目录与 §1/§3/§4 的组织描述。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.20781)
