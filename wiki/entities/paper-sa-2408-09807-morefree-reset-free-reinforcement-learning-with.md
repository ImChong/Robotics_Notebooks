---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2408.09807"
venue: "arXiv 2024"
summary: "World models increase autonomy in RL."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2408_09807_morefree-reset-free-reinforcement-learni.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# MoReFree

**MoReFree: Reset-free Reinforcement Learning with World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 261/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World models increase autonomy in RL.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World models increase autonomy in RL.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    PEG["PEG 骨干：RSSM 世界模型 + 目标条件策略 πG + 探索策略 πE + 动力学距离奖励"] --> LOOP
    LOOP["无重置在线循环"] --> R{"随机数 r < α ?"}
    R -- "是" --> BF["Back-and-Forth Go-Explore：采样评估目标 g* 与初始目标 g0"]
    BF --> FWDGO["Go 到 g*（Forward 轨迹）+ Explore"]
    FWDGO --> BACKGO["从终点继续 Go 到 g0（Back 轨迹）+ Explore"]
    R -- "否" --> EXPL["从探索目标分布 ρE 采样目标，执行 Go-Explore"]
    FWDGO --> BUF["数据加入回放缓冲 D"]
    BACKGO --> BUF
    EXPL --> BUF
    BUF --> WM["用 D 更新世界模型"]
    WM --> PE["在想象中更新 πE"]
    WM --> PG["在想象中更新 πG：目标按 α/2 取评估目标、α/2 取初始状态、1-α 取缓冲状态"]
    PE --> LOOP
    PG --> LOOP
```

图依据：论文 §3.2 Model-based RL setup（Algorithm 1）、§4.1 Back-and-Forth Go-Explore（Algorithm 2、3）、§4.2 想象中学习相关目标。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 261/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2408.09807> |
| 项目页 | <https://sites.google.com/view/morefree> |

## 核心机制（归纳）

### 策展导读要点

World models increase autonomy in RL.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2408.09807) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「MoReFree」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **73 Offline & Generalization**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2408_09807_morefree-reset-free-reinforcement-learni.md`](../../sources/papers/sun_awesome_wm_2408_09807_morefree-reset-free-reinforcement-learni.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2408.09807>
- [arXiv PDF](https://arxiv.org/pdf/2408.09807)：已通读方法部分（实验与附录以略读为准）：§3 预备知识、§4 方法（Back-and-Forth Go-Explore、想象中学习相关目标、实现细节）；§5 实验略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2408.09807)
