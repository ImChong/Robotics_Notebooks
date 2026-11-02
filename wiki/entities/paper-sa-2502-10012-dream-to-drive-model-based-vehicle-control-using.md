---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2502.10012"
venue: "arXiv 2025"
summary: "Model-based vehicle control using analytic world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2502_10012_dream-to-drive-model-based-vehicle-contr.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Dream to Drive

**Dream to Drive: Model-Based Vehicle Control Using Analytic World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 411/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Model-based vehicle control using analytic world models.（arXiv 当前版本题为《Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models》；本页沿用清单所列题名）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Model-based vehicle control using analytic world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  IN[输入: 其他交通参与者 / 路网点 / 红绿灯 / 自车速度 / 路线特征] --> ENC[场景编码器: 融合多模态特征]
  ENC --> REC[循环模块: 沿时间演化状态特征]
  REC --> H1[策略头 π_θ: 用 APG 训练, 损失式 1]
  REC --> H2[相对里程计头 f_O: 预测 s_t+1 − s_t, 损失式 3]
  REC --> H3[最优规划头 f_P: 预测下一状态偏移, 损失式 4]
  REC --> H4[逆最优状态头 f_I: 预测 s̃_t − s_t, 损失式 5]
  H1 --> SIM[可微模拟器 Waymax: 梯度穿过动力学回传]
  H2 --> SIM
  H3 -->|经逆运动学 InvKin 得动作| SIM
  H4 --> SIM
  SIM --> LOSS[与专家日志状态 ŝ_t+1 比较的状态空间损失]
  H4 -->|预测位移范数作动作置信度| MPC
  H2 --> MPC
  H3 --> MPC[测试时 MPC: 想象多条轨迹, 按逆状态范数打分<br/>聚合 k 条最优轨迹的首个动作]
```

图依据：论文 §3.1–3.4、§4 AWM agent architecture、§4.2 MPC。注意：arXiv 2502.10012 当前版本（v2）标题为 Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models，与页面标题 Dream to Drive 不同（疑似同一论文改名），图按 v2 正文绘制；MPC 中 H2 的具体用法正文仅称“使用学到的世界模型预测器”，该箭头为推断

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 411/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2502.10012> |

## 核心机制（归纳）

### 策展导读要点

Model-based vehicle control using analytic world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2502.10012) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Dream to Drive」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2502_10012_dream-to-drive-model-based-vehicle-contr.md`](../../sources/papers/sun_awesome_wm_2502_10012_dream-to-drive-model-based-vehicle-contr.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2502.10012>
- [arXiv PDF](https://arxiv.org/pdf/2502.10012)：已通读方法部分（实验与附录以略读为准）：§3 Method、§4 AWM agent architecture 与 §4.2 MPC；PDF 为 v2，标题与页面标题不符（Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models）
- 核对（2026-11-02）：查 arXiv 摘要页版本史：v1（2025-02-14）题 Dream to Drive: Model-Based Vehicle Control Using Analytic World Models，v2（2025-11-13，现行，AAAI 2026）改题；本页题名对应 v1，图按 v2 正文绘制；已补题名变化说明；本次只核对版本史与题名，流程图及方法细节未重读，沿用此前核对。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2502.10012)
