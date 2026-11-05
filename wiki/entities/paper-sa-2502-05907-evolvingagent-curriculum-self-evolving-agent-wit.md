---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2502.05907"
venue: "arXiv 2025"
summary: "Agent autonomous evolution with continual world model for long-horizon tasks."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2502_05907_evolvingagent-curriculum-self-evolving-a.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EvolvingAgent

**EvolvingAgent: Curriculum Self-evolving Agent with Continual World Model for Long-Horizon Tasks** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 257/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Agent autonomous evolution with continual world model for long-horizon tasks.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Agent autonomous evolution with continual world model for long-horizon tasks.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["长时程任务 T 与当前状态 S"] --> B["经验驱动任务规划器 Ψplan：GPT-4o + LoRA 微调，输出子任务 gi"]
    B --> C["世界模型引导的动作控制器 Πact：RSSM 世界模型预测并选优动作序列"]
    C --> D["环境执行动作 at"]
    D --> E["自验证：视频-语言嵌入余弦相似度 ≥ σ 或步数 ≥ Tmax 则终止子任务"]
    E --> F["多模态经验池 D_MEP 写入经验"]
    F --> B
    F --> G["CL 反射器 Φreflect：课程子任务选择"]
    G --> H["课程经验选择：TD 误差 / 梯度范数 / 信息增益"]
    H --> I["更新世界模型 Mw，带重要性权重与 Fisher 正则"]
    I --> C
    F -->|"子任务失败时按标签抽取经验"| B
```

图依据：论文 §IV 概述与 Fig. 2 / Algorithm 1（规划器-控制器-反射器闭环）、§IV-A 经验驱动任务规划器、§IV-B WM 引导动作控制器、§IV-C CL 反射器（两阶段课程选择与自反思）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 257/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.05907> |

## 核心机制（归纳）

### 策展导读要点

Agent autonomous evolution with continual world model for long-horizon tasks.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.05907) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EvolvingAgent」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2502_05907_evolvingagent-curriculum-self-evolving-a.md`](../../sources/papers/sun_awesome_wm_2502_05907_evolvingagent-curriculum-self-evolving-a.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2502.05907>
- [arXiv PDF](https://arxiv.org/pdf/2502.05907)：已通读方法部分（实验与附录以略读为准）：§III 预备知识、§IV EvolvingAgent（A 至 C）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2502.05907)
