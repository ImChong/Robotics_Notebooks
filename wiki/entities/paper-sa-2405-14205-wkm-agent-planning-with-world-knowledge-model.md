---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2405.14205"
venue: "arXiv 2024"
code: https://github.com/zjunlp/WKM
summary: "Agent planning with world knowledge models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2405_14205_wkm-agent-planning-with-world-knowledge.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WKM

**WKM: Agent Planning with World Knowledge Model** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 512/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Agent planning with world knowledge models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Agent planning with world knowledge models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph TR["训练阶段"]
    A["专家轨迹 τw + 经验智能体采样的失败轨迹 τl"] --> B["(a) 任务知识合成：agent 自身对比 τw/τl 生成 κ"]
    A --> C["(b) 状态知识总结：每步生成 st"]
    C --> D["状态知识库 B：(a_t, s_t, a_t+1) 三元组"]
    B --> E["(c) 训练：agent 模型学习 κ → 动作；知识模型学习 κ 与 st（LoRA 共享主干）"]
    C --> E
  end
  subgraph PL["规划阶段"]
    F["任务指令 u"] --> G["知识模型生成任务知识 κ"]
    G --> H["agent 模型规划（历史含 κ）"]
    H --> I["知识模型生成当前状态知识 st"]
    I --> J["以 st 检索 B 的 N 个近邻 → p_know"]
    H --> K["agent 模型动作概率 p_agent"]
    J --> L["a_t+1 = argmax(γ·p_agent + (1-γ)·p_know)"]
    K --> L
    L --> H
  end
  E --> F
```

图依据：论文 §3.1–§3.4 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 512/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2405.14205> |
| 代码/项目 | <https://github.com/zjunlp/WKM> |

## 核心机制（归纳）

### 策展导读要点

Agent planning with world knowledge models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2405.14205) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WKM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2405_14205_wkm-agent-planning-with-world-knowledge.md`](../../sources/papers/sun_awesome_wm_2405_14205_wkm-agent-planning-with-world-knowledge.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2405.14205>
- [arXiv PDF](https://arxiv.org/pdf/2405.14205)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 预备知识、§3 方法（3.1–3.4）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2405.14205)
