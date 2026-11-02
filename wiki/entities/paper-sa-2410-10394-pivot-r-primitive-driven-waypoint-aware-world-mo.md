---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.10394"
venue: "arXiv 2024"
code: https://github.com/abliao/PIVOT-R
summary: "Primitive-driven waypoint-aware world model for robotic manipulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_10394_pivot-r-primitive-driven-waypoint-aware.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# PIVOT-R

**PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 317/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Primitive-driven waypoint-aware world model for robotic manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Primitive-driven waypoint-aware world model for robotic manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    L["语言指令 l"] --> VLM["预训练 VLM：primitive action 解析（频率 v1，最低）"]
    O["观测图像 O_t"] --> VLM
    VLM --> P["waypoint 指示 P_t = (l, 解析出的 primitive action)，经 CLIP 文本编码器"]
    P --> SP["场景预测模块 Φsp（12 层 transformer，频率 v2）：预测 waypoint 特征 F_M't"]
    OH["历史观测 O_t-h:t，经 CLIP 视觉编码器"] --> SP
    SP --> AP["动作预测模块 Φap（3 层 transformer，频率 v3，最高）"]
    OH --> AP
    ST["机器人状态 S_t-h:t，经 MLP 编码"] --> AP
    AP --> A["动作 A't = 末端位姿增量 + 夹爪开合"]
    SP -. "L_scene：与 CLIP 编码的真值 waypoint 做 L2" .-> LOSS["总损失 L = L_scene + L_act"]
    AP -. "L_act：交叉熵" .-> LOSS
```

图依据：论文 §3.1–§3.5 与 Fig.2、式 (1)–(5)；三个模块的异步执行频率见 §3.4（Asynchronous Hierarchical Executor，v1 < v2 < v3；某模块未完成时先返回上一次结果）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 317/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.10394> |
| 代码/项目 | <https://github.com/abliao/PIVOT-R> |

## 核心机制（归纳）

### 策展导读要点

Primitive-driven waypoint-aware world model for robotic manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.10394) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「PIVOT-R」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2410_10394_pivot-r-primitive-driven-waypoint-aware.md`](../../sources/papers/sun_awesome_wm_2410_10394_pivot-r-primitive-driven-waypoint-aware.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.10394>
- [arXiv PDF](https://arxiv.org/pdf/2410.10394)：已通读方法部分（实验与附录以略读为准）：§3.1 问题定义、§3.2 输入输出、§3.3 网络、§3.4 异步分层执行器、§3.5 损失

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.10394)
