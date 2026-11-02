---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.04970"
venue: "arXiv 2026"
summary: "Releases EgoProactive, a large-scale wearable-egocentric dataset for proactive procedural assistance with explicit Out-of-Plan (OOP) annotations and recovery steps; proposes a decoupled planner–interaction architecture s"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_04970_plan-watch-recover-a-benchmark-and-archi.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Plan, Watch, Recover

**Plan, Watch, Recover: A Benchmark and Architectures for Proactive Procedural Assistance** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 123/249** 篇，分组 **291 Procedural Error Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Releases EgoProactive, a large-scale wearable-egocentric dataset for proactive procedural assistance with explicit Out-of-Plan (OOP) annotations and recovery steps; proposes a decoupled planner–interaction architecture specialized for procedural state tracking and recovery injection

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Releases EgoProactive, a large-scale wearable-egocentric dataset for proactive procedural assistance with explicit Out-of-Plan (OOP) annotations and recovery steps; proposes a decoupled planner–interaction architecture specialized for procedural state tracking and recovery injection
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    Q["用户请求 q<br/>例：Help me make an espresso"] --> P0["规划器 Planner pϕ<br/>写出初始计划 P0：步骤 + 完成视觉线索<br/>（t=0 视为一次 interrupt，P-1 为空）"]
    P0 --> PS["当前计划状态 Pt-1<br/>结构化文本块，置于交互模型输入"]
    OBS["观测 ot：以 t 结尾的 8 s 视频片段<br/>每 500 ms 一次决策"] --> IM
    PS --> IM["交互模型 pθ（Watch）<br/>对照当前步骤及线索判断"]
    IM -->|"dt = Silent"| KEEP["保持沉默<br/>Pt = Pt-1（计划不变）"]
    IM -->|"dt = Interrupt，输出话语 ut"| CALL["异步调用规划器<br/>交互模型不等待，继续处理观测"]
    CALL --> UPD{"步骤完成 还是 OOP 偏离？"}
    UPD -->|"步骤完成"| ADV["推进当前步骤状态<br/>写新的完成线索"]
    UPD -->|"OOP 偏离"| REC["Recover：用恢复步骤<br/>替换剩余步骤"]
    ADV --> VAL["新状态 Pt 通过 schema 校验后<br/>替换 Pt-1"]
    REC --> VAL
    VAL --> PS
    KEEP --> PS
```

图依据：论文 §3 / §3.2 / Fig.1、Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 123/249 |
| 分组 | 291 Procedural Error Detection |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.04970> |

## 核心机制（归纳）

### 策展导读要点

Releases EgoProactive, a large-scale wearable-egocentric dataset for proactive procedural assistance with explicit Out-of-Plan (OOP) annotations and recovery steps; proposes a decoupled planner–interaction architecture specialized for procedural state tracking and recovery injection

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.04970) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **291 Procedural Error Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Plan, Watch, Recover」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **291 Procedural Error Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2606_04970_plan-watch-recover-a-benchmark-and-archi.md`](../../sources/papers/sun_awesome_ego_2606_04970_plan-watch-recover-a-benchmark-and-archi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.04970>
- [arXiv PDF](https://arxiv.org/pdf/2606.04970)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2.1 任务与标注、§3 方法（§3.1–3.3，Fig.1/Fig.2）；其余略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.04970)
