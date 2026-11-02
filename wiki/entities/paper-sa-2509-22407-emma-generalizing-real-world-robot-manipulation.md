---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.22407"
venue: "arXiv 2025"
summary: "Generalizes real-world robot manipulation via generative visual transfer."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_22407_emma-generalizing-real-world-robot-manip.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EMMA

**EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 289/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generalizes real-world robot manipulation via generative visual transfer.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Generalizes real-world robot manipulation via generative visual transfer.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  D["真实或仿真示教视频 多视角深度视频 + 文本提示"] --> DT["DreamTransfer 双分支 DiT 主分支去噪 + ControlNet 深度结构引导"]
  DT --> GV["多视角一致的生成视频 可编辑前景 背景 光照"]
  GV --> F["视频质量过滤器 深度一致性 多视角一致性 文本-视频相似度"]
  F -->|"不达标 采样权重置 0"| X["丢弃"]
  F -->|"达标"| M["真实数据 + 高质量生成数据 均匀采样训练 VLA"]
  M --> S["对训练样本计算动作 MSE 平滑度 关节限位 合成得分 si"]
  S --> A["AdaMix 采样权重 p(i) 正比于 γ + λ(1 - si) 上调困难样本"]
  A --> V["增量训练 VLA 策略 π0 或 π0.5"]
```

图依据：论文 §2.1 / Fig.2（EMMA 总览）、§2.2 / Fig.3（DreamTransfer）、§2.3（AdaMix，式 3–7）、§3.2（基座策略 π0 / π0.5）。论文 PDF 标题为 “EMMA: Enhancing Robotic Control Robustness through a Generative Visual Model”，与本页标题不同，但 arXiv 号、项目页与方法（EMMA / DreamTransfer / AdaMix）一致。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 289/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.22407> |
| 项目页 | <https://emma-gigaai.github.io> |

## 核心机制（归纳）

### 策展导读要点

Generalizes real-world robot manipulation via generative visual transfer.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.22407) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EMMA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2509_22407_emma-generalizing-real-world-robot-manip.md`](../../sources/papers/sun_awesome_wm_2509_22407_emma-generalizing-real-world-robot-manip.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.22407>
- [arXiv PDF](https://arxiv.org/pdf/2509.22407)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 方法（2.1–2.3）、§3 实验设定与 §3.2 真机部分；附录与仿真实验略读。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.22407)
