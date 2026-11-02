---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2601.21282"
venue: "arXiv 2025"
summary: "Provides a more fine‑grained and scalable framework through concept‑specific evaluation, rigorously assessing the physical reasoning capabilities of video generation and world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2601_21282_worldbench-disambiguating-physics-for-di.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WorldBench

**WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 050/571** 篇，分组 **115 Benchmarks for World Model Simulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Provides a more fine‑grained and scalable framework through concept‑specific evaluation, rigorously assessing the physical reasoning capabilities of video generation and world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Provides a more fine‑grained and scalable framework through concept‑specific evaluation, rigorously assessing the physical reasoning capabilities of video generation and world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    GEN["数据生成：Kubric（底层 PyBullet + Blender）<br/>输出 RGB、深度、法线、分割、光流等"] --> IN["初始帧 / 短输入视频"]
    IN --> WFM["世界基础模型（WFM）/ 图生视频模型<br/>补全生成后续视频"]
    WFM --> V["生成视频"]
    subgraph S1["子集1：直觉物理（Physical Principles）"]
        GT1["真值分割 → 首帧各物体 bounding box"] --> SAM["SAM2 以 box 提示，逐帧传播物体掩码"]
        V --> SAM
        SAM --> M1["与真值掩码比较：前景 mIoU<br/>背景 RMSE（衡量背景保持）"]
    end
    subgraph S2["子集2：物理参数估计（重力 / 摩擦 / 粘度）"]
        CB["棋盘格标定：相机内参 + 每段视频外参"] --> POS["SAM2 跟踪物体（人工选提示点）<br/>掩码质心 + 恒定深度 → 3D 位置"]
        V --> POS
        POS --> FIT["曲线拟合：二次拟合得加速度<br/>线性拟合斜率得终端速度"]
        FIT --> PAR["换算物理参数：g、摩擦系数、粘度<br/>与真值或可接受范围比较"]
    end
    V --> S1
    V --> S2
```

图依据：论文 §3（Benchmark）、§3.1.1 与 §3.2.1（两个子集的 Evaluation Methodology）、Fig.1、Fig.2；本图画的是基准的评测流程，不是被测模型的结构。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 050/571 |
| 分组 | 115 Benchmarks for World Model Simulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2601.21282> |

## 核心机制（归纳）

### 策展导读要点

Provides a more fine‑grained and scalable framework through concept‑specific evaluation, rigorously assessing the physical reasoning capabilities of video generation and world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2601.21282) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **115 Benchmarks for World Model Simulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WorldBench」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **115 Benchmarks for World Model Simulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2601_21282_worldbench-disambiguating-physics-for-di.md`](../../sources/papers/sun_awesome_wm_2601_21282_worldbench-disambiguating-physics-for-di.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2601.21282>
- [arXiv PDF](https://arxiv.org/pdf/2601.21282)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Benchmark（3.1-3.2，含两处 Evaluation Methodology）。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2601.21282)
