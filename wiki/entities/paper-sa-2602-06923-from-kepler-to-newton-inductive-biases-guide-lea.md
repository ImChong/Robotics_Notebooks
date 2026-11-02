---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.06923"
venue: "arXiv 2026"
summary: "Shows how inductive biases guide transformers to learn world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_06923_from-kepler-to-newton-inductive-biases-g.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# From Kepler to Newton

**From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 550/571** 篇，分组 **Theory Interpretability & Positions**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Shows how inductive biases guide transformers to learn world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Shows how inductive biases guide transformers to learn world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  A["行星轨迹 (x, y)，Δt 采样，自回归预测下一位置"] --> B["基线（Vafa et al. 2025）：坐标分箱 tokenization + 交叉熵 NTP"]
  B --> C["失败：随机初始化 token 嵌入，空间地图弱、不平滑"]
  C --> D["偏置 1 空间平滑：连续坐标 + 回归（MSE）"]
  D --> E["新问题：回归自回归误差累积、发散"]
  E --> F["偏置 2 空间稳定：训练时给历史输入加高斯噪声（noisy context learning）"]
  F --> G["偏置 3 时间局部：改变上下文长度"]
  G -->|"上下文很短（如 2）"| H["学到 Newtonian 世界模型：线性探针可读出力 F、Fx、Fy"]
  G -->|"上下文很长（如 100）"| I["学到 Keplerian 世界模型：线性探针可读出椭圆参数 a、b、LRL 向量"]
```

图依据：论文 §2–§4 / Fig. 1（视觉摘要）、Fig. 4–6；各偏置的先后次序按论文章节叙述组织。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 550/571 |
| 分组 | Theory Interpretability & Positions |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.06923> |

## 核心机制（归纳）

### 策展导读要点

Shows how inductive biases guide transformers to learn world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.06923) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **Theory Interpretability & Positions** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「From Kepler to Newton」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Theory Interpretability & Positions**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_06923_from-kepler-to-newton-inductive-biases-g.md`](../../sources/papers/sun_awesome_wm_2602_06923_from-kepler-to-newton-inductive-biases-g.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.06923>
- [arXiv PDF](https://arxiv.org/pdf/2602.06923)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§2 偏置 1、§3 偏置 2、§4 偏置 3（含 Fig. 1、4–6 与相关正文）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.06923)
