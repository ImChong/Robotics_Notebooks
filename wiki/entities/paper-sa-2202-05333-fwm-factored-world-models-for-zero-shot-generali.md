---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2202.05333"
venue: "arXiv 2022"
summary: "Factored world models for zero-shot generalization."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2202_05333_fwm-factored-world-models-for-zero-shot.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# FWM

**FWM: Factored World Models for Zero-Shot Generalization in Robotic Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 295/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Factored world models for zero-shot generalization.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Factored world models for zero-shot generalization.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["图像观测 + 目标检测与跟踪：因子化状态 s = s_1..s_K（每个物体一张裁剪图）"] --> B["编码器 f_φ：逐物体独立编码为隐因子 z_i"]
    B --> C["隐空间转移模型 g_θ：残差 GNN 堆叠，动作 a 注入节点与边网络"]
    C --> D["预测下一隐状态 ẑ = z + g_θ(z, a)"]
    D --> E["单步对比损失训练：拉近预测与真实下一隐状态，推远随机负样本"]
    D --> H["启发式搜索：在 100×100 网格上评估约 1 万个 pick 或 place 动作"]
    G["单个目标状态示例的编码 z_g"] --> H
    H --> I["Hungarian 匹配物体并按距离、in-hand 分类器 j_ψ 给出启发式值"]
    I --> J["20 个模型的集成：对各成员启发式取平均"]
    J --> K["选择最优动作并闭环在线执行"]
```

图依据：论文 §III-A / §III-B / §III-C、Algorithm 1、§IV-D（20 模型集成、1 万个动作网格）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 295/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2022 |
| 论文 | <https://arxiv.org/abs/2202.05333> |

## 核心机制（归纳）

### 策展导读要点

Factored world models for zero-shot generalization.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2202.05333) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「FWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2202_05333_fwm-factored-world-models-for-zero-shot.md`](../../sources/papers/sun_awesome_wm_2202_05333_fwm-factored-world-models-for-zero-shot.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2202.05333>
- [arXiv PDF](https://arxiv.org/pdf/2202.05333)：已通读方法部分（实验与附录以略读为准）：§III 因子化世界模型（A 模型结构、B 对比损失训练、C 规划）、§IV-D 规划设置

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2202.05333)
