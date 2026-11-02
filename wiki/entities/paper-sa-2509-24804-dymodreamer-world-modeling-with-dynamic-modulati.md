---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.24804"
venue: "arXiv 2025"
code: https://github.com/Ultraman-Tiga1/DyMoDreamer
summary: "World modeling with dynamic modulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_24804_dymodreamer-world-modeling-with-dynamic.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DyMoDreamer

**DyMoDreamer: World Modeling with Dynamic Modulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 242/571** 篇，分组 **72 Continuous Control**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World modeling with dynamic modulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World modeling with dynamic modulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["观测 o_t（图像）"] --> DIFF["帧间差分：与 o_(t-k) 比较，阈值 ε 二值化<br/>卷积扩张周围区域得到差分掩码 M(o_t)"]
    DIFF --> OD["差分观测 o'_t = M(o_t) · o_t"]
    OD --> DENC["动态编码器 q(d_t | h_t, o'_t)<br/>得到动态调制器 d_t"]
    OBS --> SENC["随机编码器 q(z_t | h_t, o_t)<br/>得到 z_t"]
    H["RSSM 循环状态 h_t<br/>GRU：f(h_(t-1), z_(t-1), d_(t-1), a_(t-1))"] --> SENC
    H --> DENC
    SENC --> DEC["解码器 p(ô_t | h_t, z_t, d_t)"]
    DENC --> DEC
    H --> PRED["MLP 预测：ẑ_t、d̂_t、奖励 r̂_t、继续标志 ĉ_t"]
    SENC --> PRED
    DENC --> PRED
    DEC --> LOSS["端到端损失<br/>L_pred + ω_dyn·L_dyn + ω_rep·L_rep + L_reg"]
    PRED --> LOSS
    SENC --> STATE
    DENC --> STATE
    H --> STATE["模型状态 s_t = {h_t, z_t, d_t}"]
    STATE --> AC["在想象轨迹上训练 Actor-Critic<br/>（其余设置同 DreamerV3）"]
```

图依据：论文 §2.1–§2.5、式 (1)–(10)、Fig.2 总览。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 242/571 |
| 分组 | 72 Continuous Control |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.24804> |
| 代码/项目 | <https://github.com/Ultraman-Tiga1/DyMoDreamer> |

## 核心机制（归纳）

### 策展导读要点

World modeling with dynamic modulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.24804) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **72 Continuous Control** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DyMoDreamer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **72 Continuous Control**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2509_24804_dymodreamer-world-modeling-with-dynamic.md`](../../sources/papers/sun_awesome_wm_2509_24804_dymodreamer-world-modeling-with-dynamic.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.24804>
- [arXiv PDF](https://arxiv.org/pdf/2509.24804)：已通读方法部分（实验与附录以略读为准）：§2.1–§2.5 方法

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.24804)
