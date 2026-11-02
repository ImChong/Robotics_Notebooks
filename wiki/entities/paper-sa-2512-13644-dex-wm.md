---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2512.13644"
venue: "arXiv 2025"
summary: "Dexterous world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2512_13644_dex-wm.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Dex-WM

**Dex-WM** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 282/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Dexterous world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Dexterous world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  IMG["第一视角 RGB 观测 I"] --> ENC["冻结 DINOv2 编码器\npatch 级特征作为潜在状态 s"]
  HIST["历史潜在状态 s_k0 … s_kn"] --> PRED
  ENC --> HIST
  ACT["动作 a: 相邻时刻 21 点 MANO 手部关键点差\n(左右手, 同一参考系)\n+ 相机平移 / 朝向变化"] --> PRED["预测器 f_θ: Conditional Diffusion Transformer (CDiT)\nAdaLN 注入动作, 直接回归下一状态 (无迭代去噪)"]
  PRED --> NEXT["下一潜在状态 ŝ"]
  NEXT --> LS["L_state: 与真值 DINOv2 嵌入的 MSE"]
  NEXT --> HM["热图预测头 g_θ: 指尖 / 手腕热图"]
  HM --> LHC["手部一致性损失 L_HC (λ=100)"]
  NEXT -. 自回归多步预测 .-> PRED
  NEXT --> PLAN["规划: CEM 优化机器人关节角 Θ\n正运动学 G 得到动作, 最小化 C = C_state + μ·C_kp"]
  PLAN --> GOAL["到达目标状态 s_g 的零样本灵巧操作"]
```

图依据：论文 §3（问题定义、§3.1 状态与动作、§3.2 预测器、§3.3 损失、§3.4 规划）/ Fig.2–4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 282/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2512.13644> |
| 项目页 | <https://raktimgg.github.io/dexwm/> |

## 核心机制（归纳）

### 策展导读要点

Dexterous world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2512.13644) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Dex-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2512_13644_dex-wm.md`](../../sources/papers/sun_awesome_wm_2512_13644_dex-wm.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2512.13644>
- [arXiv PDF](https://arxiv.org/pdf/2512.13644)：已通读方法部分（实验与附录以略读为准）：摘要、§3.1–§3.4（规划代价 C_kp 细节与附录 CEM 参数未细读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2512.13644)
