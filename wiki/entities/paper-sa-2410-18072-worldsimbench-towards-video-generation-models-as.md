---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.18072"
venue: "arXiv 2024"
summary: "A benchmark for evaluating video generation models as world simulators."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_18072_worldsimbench-towards-video-generation-m.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WorldSimBench

**WorldSimBench: Towards Video Generation Models as World Simulators** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 052/571** 篇，分组 **115 Benchmarks for World Model Simulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A benchmark for evaluating video generation models as world simulators.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A benchmark for evaluating video generation models as world simulators.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph EPE["Explicit Perceptual Evaluation (§4.1)"]
        E1["分层评测维度 + 互联网视频字幕<br/>(OE / AD / RM 三类场景)"] --> E2["ChatGPT 扩展 meta-prompt<br/>人工筛选 → Task Instruction Prompt List"]
        E2 --> E3["各场景训练视频生成模型<br/>生成指令跟随的具身视频"]
        E3 --> E4["细粒度人类反馈标注<br/>分维度打分 + 原因"]
        E4 --> E5["HF-Embodied 数据集<br/>35,701 条 (视频, 指令, 多维分数, 原因)"]
        E5 --> E6["Human Preference Evaluator<br/>LoRA 微调 VideoLLM (Flash-VStream)"]
        E7["待评 World Simulator 按指令生成视频"] --> E6
        E6 --> E8["各维度 1~n 分，取平均"]
    end
    subgraph IME["Implicit Manipulative Evaluation (§4.2)"]
        I1["具身任务分解为可执行子任务"] --> I2["World Simulator 依据指令 + 实时观测<br/>生成预测视频"]
        I2 --> I3["视频到动作模型<br/>(预训练 IDM 或 goal-based policy)"]
        I3 --> I4["在仿真中执行动作<br/>(Minecraft / CARLA / CALVIN)"]
        I4 -->|"固定时间步后重新采样预测视频"| I2
        I4 --> I5["仿真监视器统计任务成功率 / 指标"]
    end
```

图依据：论文 §4.1 Explicit Perceptual Evaluation（Fig. 2、§4.1.2～4.1.5）、§4.2 Implicit Manipulative Evaluation（Fig. 3）；本文为基准论文，图画的是其评测流程而非被测模型结构。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 052/571 |
| 分组 | 115 Benchmarks for World Model Simulation |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.18072> |

## 核心机制（归纳）

### 策展导读要点

A benchmark for evaluating video generation models as world simulators.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.18072) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **115 Benchmarks for World Model Simulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WorldSimBench」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2410_18072_worldsimbench-towards-video-generation-m.md`](../../sources/papers/sun_awesome_wm_2410_18072_worldsimbench-towards-video-generation-m.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.18072>
- [arXiv PDF](https://arxiv.org/pdf/2410.18072)：已通读方法部分（实验与附录以略读为准）：§3～4（WorldSimBench 构建、§4.1 显式感知评测、§4.2 隐式操作评测）与 §5.1 实验设置。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.18072)
