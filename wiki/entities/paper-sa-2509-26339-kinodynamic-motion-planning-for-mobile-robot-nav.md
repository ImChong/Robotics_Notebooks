---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.26339"
venue: "arXiv 2025"
summary: "Kinodynamic motion planning across inconsistent world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_26339_kinodynamic-motion-planning-for-mobile-r.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent W...

**Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 348/571** 篇，分组 **813 Navigation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Kinodynamic motion planning across inconsistent world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Kinodynamic motion planning across inconsistent world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    BASE["KEASL 单假设（SH）搜索<br/>运动基元扩展 → 碰撞检查 → 速度约束"] --> MW["多世界假设搜索空间<br/>每个节点按世界假设各存一份扩展历史<br/>并记录分歧点"]
    MW --> VEH["VEH：扩展边须在每个假设中都有效<br/>（基线，规划偏保守）"]
    MW --> PEH["PEH：边只需在至少一个假设有效<br/>无效假设用 Rerouter 子搜索绕障，代价取平均<br/>（规划耗时长）"]
    MW --> GEH["GEH：延迟到边扩展至目标后<br/>对每个假设从首个分歧点 Reroute 到目标<br/>更新目标边代价并将目标节点重新入 OPEN"]
    GEH -->|"扩展集中在目标附近"| GEGRH["GEGRH：在 GEH 上增加图修订<br/>找到所有假设的首个分歧节点 nd<br/>ne.backPointer = nd.backPointer<br/>移除 nd 的后代，标记 nd 无效"]
    GEGRH --> SRCH["搜索继续<br/>（A* 比较 f 值，ARA* 则降低启发式膨胀）"]
    VEH --> OUT["最终解须在最新（主）世界假设中有效"]
    PEH --> OUT
    SRCH --> OUT
```

图依据：论文 §III-A 至 §III-E、Algorithm 1–5；GEH 到 GEGRH 的箭头按 §III-D/E 对 GEH 缺陷的叙述归纳。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 348/571 |
| 分组 | 813 Navigation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.26339> |

## 核心机制（归纳）

### 策展导读要点

Kinodynamic motion planning across inconsistent world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.26339) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **813 Navigation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent W...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2509_26339_kinodynamic-motion-planning-for-mobile-r.md`](../../sources/papers/sun_awesome_wm_2509_26339_kinodynamic-motion-planning-for-mobile-r.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.26339>
- [arXiv PDF](https://arxiv.org/pdf/2509.26339)：已通读方法部分（实验与附录以略读为准）：§III 技术方案（A–E）、Algorithm 1–5

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.26339)
