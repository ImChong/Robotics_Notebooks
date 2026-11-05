---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2504.15785"
venue: "arXiv 2025"
code: https://github.com/elated-sawyer/WALL-E
summary: "World alignment by neuro-symbolic learning improves world model-based LLM agents."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2504_15785_wall-e-2-0-world-alignment-by-neurosymbo.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WALL-E 2.0

**WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 511/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World alignment by neuro-symbolic learning improves world model-based LLM agents.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World alignment by neuro-symbolic learning improves world model-based LLM agents.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["当前观测 o_t"] --> AGENT["LLM Agent 提出动作 a_t"]
    AGENT --> WM["LLM 世界模型预测 ô_t+1<br/>(判断该转移成功/失败)"]
    CR["代码规则集 R_code<br/>(含 Knowledge Graph / Scene Graph 参数)"] --> MAP["MapExecute：激活的规则校验预测<br/>不一致时规则输出覆盖 LLM 预测"]
    WM --> MAP
    MAP --> FLAG{"flag 为 True？"}
    FLAG -->|"否：返回 feedback / suggestion"| AGENT
    FLAG -->|"是：动作被接受（或达 replan 上限）"| ENV["在环境中执行 a_t，得到真实 o_t+1"]
    ENV --> TRAJ["累积真实轨迹 τ_real 与预测轨迹 τ_pred"]
    TRAJ --> S1["Stage 1 比较两条轨迹<br/>划分预测正确 / 错误的转移"]
    S1 --> S2["Stage 2 LLM 归纳符号知识<br/>Action Rules、Knowledge Graph、Scene Graph"]
    S2 --> S3["Stage 3 LLM 将符号知识翻译为 Python 代码规则"]
    S3 --> S4["Stage 4 最大覆盖贪心剪枝代码规则"]
    S4 --> CR
    ENV -->|"下一步"| OBS
```

图依据：论文 §3.1（Stage 1–4）、§3.2 / Fig.2–3 / Algorithm 1–3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 511/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2504.15785> |
| 代码/项目 | <https://github.com/elated-sawyer/WALL-E> |

## 核心机制（归纳）

### 策展导读要点

World alignment by neuro-symbolic learning improves world model-based LLM agents.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.15785) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WALL-E 2.0」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **83 GUI / Computer-Use Agents**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2504_15785_wall-e-2-0-world-alignment-by-neurosymbo.md`](../../sources/papers/sun_awesome_wm_2504_15785_wall-e-2-0-world-alignment-by-neurosymbo.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2504.15785>
- [arXiv PDF](https://arxiv.org/pdf/2504.15785)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3 全节（3.1.1–3.1.4、3.2）、Algorithm 1–3、Fig.2–3

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2504.15785)
