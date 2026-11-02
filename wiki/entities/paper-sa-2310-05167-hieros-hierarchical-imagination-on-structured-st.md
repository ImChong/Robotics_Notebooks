---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2310.05167"
venue: "arXiv 2023"
summary: "Hierarchical imagination on structured state space sequence world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2310_05167_hieros-hierarchical-imagination-on-struc.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Hieros

**Hieros: Hierarchical Imagination on Structured State Space Sequence World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 244/571** 篇，分组 **72 Continuous Control**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Hierarchical imagination on structured state space sequence world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Hierarchical imagination on structured state space sequence world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    ENV["环境<br/>观测 o / 奖励 r"] --> L0
    subgraph L0["subactor 0 最低层 (唯一与环境交互)"]
        WM0["S5 世界模型 S5WM<br/>h_t 确定性 + z_t 随机"] --> AC0["actor-critic<br/>在想象 rollout 上训练"]
        WM0 --> AE0["subgoal autoencoder<br/>压缩 h_t 为离散子目标"]
    end
    AC0 -->|"动作 a"| ENV
    L0 -->|"连续 k 个世界模型状态作为观测"| L1
    subgraph L1["subactor 1 更高层"]
        WM1["S5WM"] --> AC1["actor-critic"]
        WM1 --> AE1["subgoal autoencoder"]
    end
    AC1 -->|"子目标 g, 每 k 步更新"| AC0
    AE0 -.->|"子目标奖励 r_g 与新颖性奖励 r_nov"| AC0
    L1 -.->|"更高层同理, 可继续堆叠"| LN["subactor i ..."]
    ENV -.-> BUF["经验数据集<br/>ETBS 时间均衡 O(1) 采样"]
    BUF -.-> WM0
```

图依据：论文 §2.1-§2.3 / Fig.1（层级结构）；奖励 r = r_extr + w_g·r_g + w_nov·r_nov（Eq.1）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 244/571 |
| 分组 | 72 Continuous Control |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2310.05167> |

## 核心机制（归纳）

### 策展导读要点

Hierarchical imagination on structured state space sequence world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2310.05167) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **72 Continuous Control** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Hieros」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2310_05167_hieros-hierarchical-imagination-on-struc.md`](../../sources/papers/sun_awesome_wm_2310_05167_hieros-hierarchical-imagination-on-struc.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2310.05167>
- [arXiv PDF](https://arxiv.org/pdf/2310.05167)：已通读方法部分（实验与附录以略读为准）：§1 引言（贡献）、§2 Methodology（2.1 多层层级想象、2.2 S5WM、2.3 ETBS）；§3 实验、附录略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2310.05167)
