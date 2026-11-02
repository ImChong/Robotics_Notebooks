---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.03298"
venue: "arXiv 2025"
summary: "Builds structured world models with object-centric representations."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_03298_dyn-o-building-structured-world-models-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Dyn-O

**Dyn-O: Building Structured World Models with Object-Centric Representations** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 126/571** 篇，分组 **45 Object-Centric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Builds structured world models with object-centric representations.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Builds structured world models with object-centric representations.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph P1["阶段1 物体中心表征学习（Cosmos 编解码器冻结）"]
        direction TB
        O["图像观测 o"] --> CE["Cosmos Encoder<br/>patch 特征 f"]
        CE --> SA["Slot Attention<br/>K 个 slot 竞争绑定物体"]
        SAM["SAM2 分割掩码 m<br/>作为 attention mask"] -.->|"概率按 log 调度退火至不用"| SA
        SA --> Z["物体 slot z = [z1..zK]"]
        Z --> SD["Slot-Dec 重建特征 f̂"]
        SD --> CD["Cosmos Decoder 重建 ô"]
    end
    Z --> DYN
    subgraph P2["阶段2 动力学学习"]
        direction TB
        DYN["slot 间 Self-Attn（无位置编码，含动作 a_t）"] --> SSM["共享 SSM 更新每个 slot 的隐状态 h"]
        SSM --> PZ["Pred：下一步 slot ẑ_t+1"]
        SSM --> PR["Cross-Attn 查询 r_cls：奖励 r̂_t"]
        SSM --> PD["Cross-Attn 查询 d_cls：终止 d̂_t"]
    end
    subgraph P3["静动态解耦（可选，与世界模型联合优化）"]
        direction TB
        Z2["slot z"] --> MC["Mc → 静态特征 c<br/>时间不变性 + InfoNCE 对比"]
        Z2 --> MV["Mv → 动态特征 v"]
        MC --> MZ["Mz 重建 ẑ（c 停梯度）"]
        MV --> MZ
        MC --> DISC["判别器 Disc（Wasserstein + LeCam）<br/>对抗降低 c 与 v 的互信息"]
        MV --> DISC
    end
    Z --> Z2
```

图依据：论文 §3 / §3.1 / §3.2、Fig.2（表征学习与动力学学习）、Fig.3（静动态解耦）、Alg.1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 126/571 |
| 分组 | 45 Object-Centric World Models |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.03298> |

## 核心机制（归纳）

### 策展导读要点

Builds structured world models with object-centric representations.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.03298) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **45 Object-Centric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Dyn-O」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **45 Object-Centric World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2507_03298_dyn-o-building-structured-world-models-w.md`](../../sources/papers/sun_awesome_wm_2507_03298_dyn-o-building-structured-world-models-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.03298>
- [arXiv PDF](https://arxiv.org/pdf/2507.03298)：已通读方法部分（实验与附录以略读为准）：§3 Method（§3.1–3.2，含静动态解耦与 Alg.1）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.03298)
