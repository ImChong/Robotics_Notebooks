---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.02393"
venue: "arXiv 2026"
summary: "Extends interactive world models to 1000‑frame horizons through pose‑free hierarchical memory."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_02393_infinite-world-scaling-interactive-world.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Infinite-World

**Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 032/571** 篇，分组 **113 Interactive World Models & Game Engines**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Extends interactive world models to 1000‑frame horizons through pose‑free hierarchical memory.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Extends interactive world models to 1000‑frame horizons through pose‑free hierarchical memory.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["训练数据策略（§3.3）"]
        PT["阶段1 开放域预训练<br/>网络采集的短视频，学视觉先验与局部动力学"] --> FT["阶段2 Revisit-Dense Dataset 微调<br/>自采、重访密集的长视频，激活空间记忆"]
    end
    V["真实视频序列"] --> PE["VGGT 估计相邻帧相机位姿变化 ΔP"]
    PE --> DEC["解耦为平移幅度 / 旋转幅度"]
    DEC --> TRI["三态标注（阈值 τ1, τ2）：<br/>No-operation / Discrete Action / Uncertain"]
    TRI --> AE["Action Encoder<br/>两层 stride-2 1D 卷积，4× 下采样 → e_act"]
    V --> VAE["VAE 编码历史帧 → 历史潜变量 z_1:L"]
    VAE --> HP{"上下文长度 L ≤ k·T_max ?"}
    HP -->|"是：直接压缩"| F1["时间编码器 f_φ（k=4）→ z_com"]
    HP -->|"否：层次压缩"| LC["滑窗切成重叠 chunk<br/>每个 chunk 局部压缩"]
    LC --> GC["拼接后再经 f_φ 全局压缩 → z_com"]
    F1 --> CAT["拼接：z_com + 末帧潜变量 z_loc + 加噪目标 z_t + 掩码 m"]
    GC --> CAT
    CAT --> ADD["逐元素加上 e_act"]
    AE --> ADD
    ADD --> DIT["DiT 块（Self-Attn / Cross-Attn / FFN）× n"]
    DIT --> OUT["预测目标视频潜变量 → 生成视频"]
    DIT -.->|"生成损失端到端联合优化 f_φ，无需位姿元数据"| F1
    FT -.-> DIT
```

图依据：论文 §3.1-3.3（HPMC、不确定性感知动作标注、重访密集微调）、Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 032/571 |
| 分组 | 113 Interactive World Models & Game Engines |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.02393> |

## 核心机制（归纳）

### 策展导读要点

Extends interactive world models to 1000‑frame horizons through pose‑free hierarchical memory.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.02393) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **113 Interactive World Models & Game Engines** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Infinite-World」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **113 Interactive World Models & Game Engines**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_02393_infinite-world-scaling-interactive-world.md`](../../sources/papers/sun_awesome_wm_2602_02393_infinite-world-scaling-interactive-world.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.02393>
- [arXiv PDF](https://arxiv.org/pdf/2602.02393)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 Methodology（3.1-3.3）。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.02393)
