---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2503.02247"
venue: "arXiv 2025"
summary: "Integrates VLMs into world models for object goal navigation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_02247_wmnav-integrating-vision-language-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WMNav

**WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 357/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Integrates VLMs into world models for object goal navigation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Integrates VLMs into world models for object goal navigation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  OBS["6 个旋转角 30/90/150/210/270/330 度的 RGB-D 图像"] --> PAN["全景图"]
  PAN --> PV["PredictVLM: 为每个方向预测 0-10 的好奇值 (目标出现可能性)"]
  PV --> PJ["结合深度与位姿, 投影到俯视可导航地图 Mnav"]
  PJ --> MG["与上一步好奇值地图取最小值融合 → Curiosity Value Map"]
  MG --> MEM["存入记忆"]
  MEM --> BK["投影回当前可导航区域, 按方向取平均好奇值"]
  BK --> SEL["选择得分最高的方向 α"]
  SEL --> PL["PlanVLM: 输入该方向图像与上一步子任务 → 新子任务 + GoalFlag"]
  PL -.->|"cost: 子任务与目标标志写入提示词"| RV
  SEL --> AP["Action Proposer: 在可导航区域采样 K 个极坐标动作, 过滤已探索区域"]
  AP --> RV["ReasonVLM (探索阶段): 从带标注的候选动作中选一个"]
  RV --> EXE["执行动作"]
  EXE -.->|下一时间步| OBS
  RV -->|目标出现在视野| GA["目标接近阶段: Goal Proposer 稠密采样, GoalVLM 选目标位置"]
  GA --> STOP["距离目标 < dthres 时停止"]
```

图依据：论文 §III-B（Overview）、§III-C（World Model）、§III-D（Subtask Decomposition）、§III-E（Two-stage Action Proposer）与 Fig.2–Fig.5。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 357/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.02247> |
| 项目页 | <https://b0b8k1ng.github.io/WMNav/> |

## 核心机制（归纳）

### 策展导读要点

Integrates VLMs into world models for object goal navigation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.02247) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WMNav」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2503_02247_wmnav-integrating-vision-language-models.md`](../../sources/papers/sun_awesome_wm_2503_02247_wmnav-integrating-vision-language-models.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.02247>
- [arXiv PDF](https://arxiv.org/pdf/2503.02247)：已通读方法部分（实验与附录以略读为准）：§III 方法全部（任务定义、概览、世界模型、子任务分解、两阶段动作提议器）；§IV 实验仅略读。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.02247)
