---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2603.06445"
venue: "arXiv 2026"
summary: "First large-scale dataset for emulative simulation of mental exploration."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2603_06445_what-if-emulative-simulation-with-world.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# What if? Emulative Simulation with World Models for Situated Reasoning (Wande...

**What if? Emulative Simulation with World Models for Situated Reasoning (WanderDream)** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 185/571** 篇，分组 **52 Game & Open-World Exploration Application**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

First large-scale dataset for emulative simulation of mental exploration.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- First large-scale dataset for emulative simulation of mental exploration.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["场景: HM3D / ScanNet++ / 真实全景录制"] --> B["WanderDream-Gen: 从当前状态 s0 到目标情境 sT 的想象轨迹全景视频"]
    B --> B1["机器人: 19 类地标 object navigation, Habitat-Sim 最短路径, 离散为 MOVE_FORWARD / TURN_LEFT / TURN_RIGHT"]
    B --> B2["人类: interacting / standing / sitting; 无障碍时直接插值, 有障碍时 3D PRM + Dijkstra"]
    B1 --> C["WanderDream-QA: GPT-5 + Set-of-Mark 生成, 每视频 10 问: 起点 3 / 路径 4 / 终点 3"]
    B2 --> C
    C --> D["数据质量控制与统计 + 真实世界测试集"]
    D --> E["emulative simulation 评测框架"]
    E --> F["当前观测 s0 + 问题"]
    F --> G["MLLM 提示扩展: 把目标轨迹分解为动作序列"]
    G --> H["世界模型生成想象视频轨迹: HunyuanVideo / CogVideoX / Wan, 零样本或在 WanderDream-Gen 上微调"]
    H --> I["MLLM 沿想象轨迹推理并回答: Qwen3-VL / LLaVA-OneVision"]
    E -.对照.-> J["MindJourney 闭环: MLLM 逐步提出相机动作 + SVC 新视角合成, 仅用于真实测试集"]
```

图依据：论文 §3.1–3.3（Fig. 4–5）、§4（Fig. 6）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 185/571 |
| 分组 | 52 Game & Open-World Exploration Application |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.06445> |

## 核心机制（归纳）

### 策展导读要点

First large-scale dataset for emulative simulation of mental exploration.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.06445) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **52 Game & Open-World Exploration Application** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「What if? Emulative Simulation with World Models for Situated Reasoning (Wande...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **52 Game & Open-World Exploration Application**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2603_06445_what-if-emulative-simulation-with-world.md`](../../sources/papers/sun_awesome_wm_2603_06445_what-if-emulative-simulation-with-world.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2603.06445>
- [arXiv PDF](https://arxiv.org/pdf/2603.06445)：已通读方法部分（实验与附录以略读为准）：§1、§3、§4、§5.1 的方法与实验设定

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2603.06445)
