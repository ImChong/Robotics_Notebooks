---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.13232"
venue: "arXiv 2024"
summary: "Learning and leveraging environment dynamics in web navigation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_13232_web-agents-with-world-models-learning-an.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Web Agents with World Models

**Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 513/571** 篇，分组 **83 GUI / Computer-Use Agents**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning and leveraging environment dynamics in web navigation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Learning and leveraging environment dynamics in web navigation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph T["世界模型训练"]
        A["Step I 采集轨迹：LLM agent 与环境交互，得到 D = {I, o_t, a_t, o_t+1}"] --> B["Step II 转移聚焦的观测抽象：Hungarian algorithm 匹配 o_t 与 o_t+1 的元素"]
        B --> C["得到 UPDATED / DELETED / ADDED 状态转移列表"]
        C --> D["LLM 转写为自由形式描述 õ_t+1，替换 o_t+1"]
        D --> E["Step III 学习环境动态：以 next-token 损失训练世界模型 φ，输入 I、o_t、a_t"]
    end
    subgraph P["推理时策略优化（策略模型 θ 冻结）"]
        F["策略模型 θ 以 top-p 采样 k 个候选动作"] --> G["世界模型 φ 为每个候选预测下一观测 õ_t+1"]
        G --> H["价值函数 V（现成 LLM）估计各候选的奖励"]
        H --> I["argmax 选出奖励最高的动作并执行"]
    end
    E --> G
```

图依据：论文 §4.1 / §4.2 与 Fig.3、Fig.5、式 (1)–(3)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 513/571 |
| 分组 | 83 GUI / Computer-Use Agents |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.13232> |

## 核心机制（归纳）

### 策展导读要点

Learning and leveraging environment dynamics in web navigation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.13232) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **83 GUI / Computer-Use Agents** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Web Agents with World Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2410_13232_web-agents-with-world-models-learning-an.md`](../../sources/papers/sun_awesome_wm_2410_13232_web-agents-with-world-models-learning-an.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.13232>
- [arXiv PDF](https://arxiv.org/pdf/2410.13232)：已通读方法部分（实验与附录以略读为准）：§3 预备分析、§4.1 世界模型训练、§4.2 推理时策略优化

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.13232)
