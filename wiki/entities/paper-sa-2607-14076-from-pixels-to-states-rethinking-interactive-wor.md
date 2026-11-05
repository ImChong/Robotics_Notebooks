---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2607.14076"
venue: "arXiv 2026"
summary: "Examines interactive game world modeling along four dimensions."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2607_14076_from-pixels-to-states-rethinking-interac.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# From Pixels to States

**From Pixels to States: Rethinking Interactive World Models as Game Engines** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 159/571** 篇，分组 **51 General Interactive Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Examines interactive game world modeling along four dimensions.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Examines interactive game world modeling along four dimensions.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    L["游戏引擎的 action-state-observation 循环（Fig. 1）<br/>玩家动作 → 按规则更新游戏状态 → 由状态渲染观测"] --> D1["① 玩家动作控制 Player Action Control"]
    L --> D2["② 游戏状态动力学 Game State Dynamics"]
    L --> D3["③ 状态-观测持久性 State-Observation Persistence"]
    L --> D4["④ 实时交互生成 Real-Time Interactive Generation"]
    D1 --> D1a["几何轨迹 / 运动信号 / 语义事件"]
    D2 --> D2a["状态纠缠于观测 / 学习到的潜状态 / 显式描述"]
    D3 --> D3a["记忆为已存观测 / 记忆为当前状态估计"]
    D4 --> D4a["降低生成延迟 / 降低条件作用延迟"]
    D2a -.-> E["§4 数据引擎（Black Myth: Wukong）"]
    E --> E1["众包玩家游戏采集<br/>引擎逐 tick 导出动作与状态 JSON + ReShade/OBS 分屏录制 RGB 与深度"]
    E1 --> E2["数据处理：时间戳逐帧对齐 → 按 boss 战切分样本 → 丢弃掉帧/卡顿/流间不一致样本"]
    E2 --> E3["槽位结构化标注：slot caption + semantic caption（Qwen3-VL-235B-A22B-Instruct）"]
```

图依据：综述分类体系图（§1 Fig. 1、§3.1–§3.4 四个维度及其家族）加数据采集流程图（§4、Fig. 2）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 159/571 |
| 分组 | 51 General Interactive Frameworks |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.14076> |

## 核心机制（归纳）

### 策展导读要点

Examines interactive game world modeling along four dimensions.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.14076) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **51 General Interactive Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「From Pixels to States」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **51 General Interactive Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2607_14076_from-pixels-to-states-rethinking-interac.md`](../../sources/papers/sun_awesome_wm_2607_14076_from-pixels-to-states-rethinking-interac.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2607.14076>
- [arXiv PDF](https://arxiv.org/pdf/2607.14076)：已通读全文正文（§1 引言、§2 相关工作、§3 四个维度、§4 数据引擎、§5 结论；参考文献略读）；流程图依据 §1 Fig. 1、§3、§4 与 Fig. 2。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2607.14076)
