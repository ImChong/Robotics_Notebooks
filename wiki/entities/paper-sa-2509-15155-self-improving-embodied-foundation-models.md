---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2509.15155"
venue: "arXiv 2025"
summary: "Self-improving embodied foundation models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2509_15155_self-improving-embodied-foundation-model.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Self-Improving Embodied Foundation Models

**Self-Improving Embodied Foundation Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 227/571** 篇，分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Self-improving embodied foundation models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Self-improving embodied foundation models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph S1["阶段一：Supervised Fine-Tuning（SFT）"]
    PT["预训练多模态基础模型（实验中为 PaLI 3B）"] --> EFM1["初始化 EFM"]
    DS["模仿学习数据集 D：(o_t, a_t, g_t')"] --> BC["目标 a：目标条件行为克隆 L_BC"]
    DS --> STG["目标 b：steps-to-go 预测 L_steps-to-go（预测距离达成目标还差多少步）"]
    EFM1 --> BC
    EFM1 --> STG
    BC --> CK1["Stage 1 checkpoint"]
    STG --> CK1
  end
  subgraph S2["阶段二：Self-Improvement（在线 RL）"]
    CK1 --> POL["初始化待优化策略"]
    CK1 --> FRZ["冻结的 Stage 1 checkpoint：用于计算奖励与成功检测"]
    FRZ --> D["d(o,g)：期望 steps-to-go"]
    D --> REW["奖励 r = d(o_t,g) − d(o_{t+1},g)"]
    D --> SUC["成功检测：d(o,g) ≤ s"]
    POL --> ROLL["采样指令 g 并执行策略"]
    ROLL --> END["结束条件：检测到成功 / 达最大步数 / 人工终止"]
    SUC --> END
    END --> BUF["计算蒙特卡洛回报 R_t，存入 replay buffer"]
    REW --> BUF
    BUF --> UPD["REINFORCE 更新 N 步，随后清空 buffer"]
    UPD --> POL
  end
```

图依据：论文 §2.1 / §2.2、Fig. 1 与 Algorithm 1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 227/571 |
| 分组 | 63 World Models for VLA Training & Evaluation Combining Paradigms |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2509.15155> |

## 核心机制（归纳）

### 策展导读要点

Self-improving embodied foundation models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.15155) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **63 World Models for VLA Training & Evaluation Combining Paradigms** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Self-Improving Embodied Foundation Models」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **63 World Models for VLA Training & Evaluation Combining Paradigms**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2509_15155_self-improving-embodied-foundation-model.md`](../../sources/papers/sun_awesome_wm_2509_15155_self-improving-embodied-foundation-model.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2509.15155>
- [arXiv PDF](https://arxiv.org/pdf/2509.15155)：已通读方法部分（§1 引言、§2 方法、§3 奖励直觉）；实验（§4）略读：实验设定与主要结果。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2509.15155)
