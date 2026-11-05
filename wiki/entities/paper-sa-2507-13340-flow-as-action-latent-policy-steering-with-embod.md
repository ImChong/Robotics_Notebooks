---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.13340"
venue: "arXiv 2025"
summary: "Latent policy steering with embodiment-agnostic pretrained world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_13340_flow-as-action-latent-policy-steering-wi.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Flow-as-Action

**Flow-as-Action: Latent Policy Steering with Embodiment-Agnostic Pretrained World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 296/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Latent policy steering with embodiment-agnostic pretrained world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Latent policy steering with embodiment-agnostic pretrained world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph PRE["阶段一：跨 embodiment 预训练（数据集 C）"]
        direction TB
        CD["多 embodiment 数据 C<br/>（仿真机器人 / 公开机器人 / 人类 play 视频）"] --> OF["计算 optical flow"]
        OF --> FE["optical flow 编码器（卷积）<br/>输出 ‖A‖ 维向量，作为 embodiment 无关动作"]
        FE --> WM0["World Model 预训练<br/>（Dreamer v3 目标与架构，不用本体感觉）"]
    end
    subgraph FT["阶段二：目标 embodiment 微调（少量专家数据 E）"]
        direction TB
        ED["目标 embodiment 专家演示 E"] --> BP["训练 base policy π（行为克隆）"]
        WM0 --> SW["把 optical flow 编码器替换为<br/>归一化机器人动作（同为 ‖A‖ 维）"]
        ED --> SW
        SW --> WM1["在 E 上用同一 Dreamer v3 目标微调 WM"]
        WM1 --> VF["鲁棒价值函数 V（梯度不回传 WM）<br/>训练样本：专家状态 + π 规划动作产生的状态<br/>相似度惩罚回到专家数据（Alg.1）"]
        BP --> VF
    end
    subgraph INF["阶段三：推理 Latent Policy Steering"]
        direction TB
        OBS["当前观察编码为潜在状态"] --> CAND["π 采样 B 条长度 h 的候选动作计划"]
        CAND --> ROLL["WM 在潜在空间预测各计划的未来状态"]
        ROLL --> SCORE["V 估值 → 折扣加权平均得计划价值"]
        SCORE --> BEST["执行价值最高的计划 h 步后重规划"]
    end
    PRE --> FT
    BP --> CAND
    WM1 --> ROLL
    VF --> SCORE
```

图依据：论文 §IV-A–§IV-C / Fig.1、Algorithm 1（论文 v4 标题为 Latent Policy Steering with Embodiment-Agnostic Pretrained World Models）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 296/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.13340> |

## 核心机制（归纳）

### 策展导读要点

Latent policy steering with embodiment-agnostic pretrained world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.13340) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Flow-as-Action」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2507_13340_flow-as-action-latent-policy-steering-wi.md`](../../sources/papers/sun_awesome_wm_2507_13340_flow-as-action-latent-policy-steering-wi.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.13340>
- [arXiv PDF](https://arxiv.org/pdf/2507.13340)：已通读方法部分（实验与附录以略读为准）：§III 预备知识、§IV 方法（§IV-A Flow-as-Action、§IV-B 目标 embodiment 微调与 Algorithm 1、§IV-C Latent Policy Steering），§V 实验设定略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.13340)
