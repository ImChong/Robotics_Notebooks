---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.20545"
venue: "arXiv 2026"
summary: "Argues that generators must maintain a continuously evolving world state rather than rendering plausible frames only at observation time."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_20545_current-world-models-lack-a-persistent-s.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Current World Models Lack a Persistent State Core

**Current World Models Lack a Persistent State Core** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 002/571** 篇，分组 **101 Memory Mechanisms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Argues that generators must maintain a continuously evolving world state rather than rendering plausible frames only at observation time.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Argues that generators must maintain a continuously evolving world state rather than rendering plausible frames only at observation time.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    N["Natural-25 提示集<br/>25 个场景族 × 四级事件设计（空间位移 vs 状态变化）"] --> REC["事件-视角记录 ri = (x0, e, τ, ν, π)<br/>初始观测 / 事件 / 视角干预 / 可见性 / 提示变体"]
    REC --> LIB["WRBenchLib：按各模型可接受的接口转换输入<br/>（相机轨迹 / 源视频 / 几何条件 / 纯文本提示）<br/>记录实际送达条件、生成视频与 provenance"]
    LIB --> VID["各生成模型输出视频"]
    VID --> D1["(i) 请求相机精度 / (ii) 提示-相机对齐<br/>VGGT-Ω 轨迹恢复 / CamAlign"]
    VID --> D3["(iii) 视觉完整性<br/>DINOv2 特征代理"]
    VID --> D4["(iv) 可见时空间与状态一致性<br/>Qwen-3.5-9B 的是非探针"]
    VID --> D5{"(v) 重观测支持？<br/>Qwen-3-VL-8B 门控：隐藏区间 + 目标返回 + 可辨识"}
    D5 -->|"满足"| D6["(vi) 重观测后空间与状态一致性"]
    D5 -->|"不满足"| NA["记为 NA（支持不足），不计入 (vi)"]
    D1 --> OUT["六维诊断画像（不合并为单一分数）"]
    D3 --> OUT
    D4 --> OUT
    D6 --> OUT
    NA --> OUT
    OUT -.-> HUM["人工偏好标注：六个维度逐轴独立标注<br/>用于校准自动评测器"]
```

图依据：论文 §3.1–3.5 / Fig. 2（基准评测流程，非生成模型方法）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 002/571 |
| 分组 | 101 Memory Mechanisms |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.20545> |

## 核心机制（归纳）

### 策展导读要点

Argues that generators must maintain a continuously evolving world state rather than rendering plausible frames only at observation time.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.20545) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **101 Memory Mechanisms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Current World Models Lack a Persistent State Core」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **101 Memory Mechanisms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_20545_current-world-models-lack-a-persistent-s.md`](../../sources/papers/sun_awesome_wm_2606_20545_current-world-models-lack-a-persistent-s.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.20545>
- [arXiv PDF](https://arxiv.org/pdf/2606.20545)：已通读方法部分（实验与附录以略读为准）：摘要、§3 WRBench 套件（3.1–3.5）；§4 实验与附录略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.20545)
