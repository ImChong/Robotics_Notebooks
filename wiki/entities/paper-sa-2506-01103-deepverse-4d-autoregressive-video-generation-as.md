---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.01103"
venue: "arXiv 2025"
summary: "4D autoregressive video generation as a world model."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_01103_deepverse-4d-autoregressive-video-genera.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DeepVerse

**DeepVerse: 4D Autoregressive Video Generation as a World Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 092/571** 篇，分组 **43 3D & 4D World Models General Purpose**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

4D autoregressive video generation as a world model.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- 4D autoregressive video generation as a world model.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["输入观测 v_0<br/>初始化记忆池 M = {ŝ_0}"] --> B["4D 状态 ŝ_t = (v_t, g_t)<br/>g_t = 深度 d_t(视差平方根) + 相机射线图 raymap"]
    B --> C["通道拼接三种模态<br/>复用预训练 VAE 潜空间"]
    M["记忆池 M<br/>历史状态对齐到初始观测坐标系"] --> S["选择机制 ψ<br/>按相机位姿检索空间相邻的历史状态"]
    T["文本动作 a_t<br/>相机位姿变化可算法生成文本"] --> F
    C --> F["自回归预测 f_θ<br/>输入: a_t、当前状态、最近 m 个状态、检索的空间条件"]
    S --> F
    F --> G["预测未来状态序列 ŝ_{t+1:t+k}<br/>写入缓存与记忆池"]
    G --> W["滑动窗口: 末尾 m 个状态作下一窗口上下文<br/>按 d_max 重新缩放后继续推理"]
    W --> F
```

图依据：论文 §2.1 Problem Formulation、§2.2 Model Components、§2.4 Long-Duration Inference / Algorithm 1、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 092/571 |
| 分组 | 43 3D & 4D World Models General Purpose |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.01103> |

## 核心机制（归纳）

### 策展导读要点

4D autoregressive video generation as a world model.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.01103) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **43 3D & 4D World Models General Purpose** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DeepVerse」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **43 3D & 4D World Models General Purpose**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_01103_deepverse-4d-autoregressive-video-genera.md`](../../sources/papers/sun_awesome_wm_2506_01103_deepverse-4d-autoregressive-video-genera.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.01103>
- [arXiv PDF](https://arxiv.org/pdf/2506.01103)：已通读方法部分（实验与附录以略读为准）：§2.1–2.4（问题形式化、模型组件、数据集构建、长时推理）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.01103)
