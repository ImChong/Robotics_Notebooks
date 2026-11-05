---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.02996"
venue: "arXiv 2025"
code: https://github.com/matthieu-perso/spatial_world_models
summary: "Linear spatial world models emerge in LLMs."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_02996_linear-spatial-world-models-emerge-in-la.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Linear Spatial World Models Emerge in Large Language Models

**Linear Spatial World Models Emerge in Large Language Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 557/571** 篇，分组 **Theory Interpretability & Positions**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Linear spatial world models emerge in LLMs.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Linear spatial world models emerge in LLMs.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["合成数据集：61 个物体 + 6 种空间关系，生成自然语言提示并标注真值位置"] --> B["LLaMA-3.2-3B-Instruct 前向传播"]
    B --> C["提取句号前最后一个 token 的残差流激活（第 8 / 12 / 24 层）"]
    C --> D["线性 probe 与非线性 MLP probe 预测空间关系 / 物体位置"]
    D --> E["PCA 降维，提取空间子空间（检验反义关系是否近似反向）"]
    E --> F["检验组合性：关系向量之和 vs 预测和（2D / 3D）"]
    E --> G["由 PCA 方向构造 steering 向量 v_r"]
    G --> H["在最后 token 处注入 h' = h + α v_r"]
    H --> I["让模型回答物体所在方向，统计各关系成功率"]
```

图依据：论文 §4.1 数据与激活提取 / §4.2 Probing / §4.3 Causal Experiments / §5.1–5.4（本文是分析型工作，图为其实验流程而非模型架构）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 557/571 |
| 分组 | Theory Interpretability & Positions |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.02996> |
| 代码/项目 | <https://github.com/matthieu-perso/spatial_world_models> |

## 核心机制（归纳）

### 策展导读要点

Linear spatial world models emerge in LLMs.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.02996) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Theory Interpretability & Positions** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Linear Spatial World Models Emerge in Large Language Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Theory Interpretability & Positions**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_02996_linear-spatial-world-models-emerge-in-la.md`](../../sources/papers/sun_awesome_wm_2506_02996_linear-spatial-world-models-emerge-in-la.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.02996>
- [arXiv PDF](https://arxiv.org/pdf/2506.02996)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 形式化定义、§4 方法（数据、probing、因果实验）、§5.1 / 5.2 / 5.4 实验流程

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.02996)
