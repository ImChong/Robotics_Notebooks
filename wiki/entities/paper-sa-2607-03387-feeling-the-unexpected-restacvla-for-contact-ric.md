---
type: entity
tags: [paper, curated-index, awesome-touch, sun254667-touch]
status: complete
updated: 2026-10-06
arxiv: "2607.03387"
venue: "arXiv 2026 (IROS 2026)"
summary: "Addresses modality collapse where high-bandwidth visual features overshadow sparse tactile cues; inspired by Predictive Coding."
related:
  - ../entities/awesome-touch.md
  - ../overview/sun-awesome-touch-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/bimanual-manipulation.md
sources:
  - ../../sources/papers/sun_awesome_touch_2607_03387_feeling-the-unexpected-restacvla-for-con.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
  - ../../sources/repos/awesome-touch.md
---

# Feeling the Unexpected

**Feeling the Unexpected: ResTacVLA for Contact-Rich Manipulation via Residual Tactile Representation** 收录于 [Awesome Touch](https://github.com/sun254667/awesome-touch) **第 003/065** 篇，分组 **21 Unified Vision-Tactile-Language-Action Frameworks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Addresses modality collapse where high-bandwidth visual features overshadow sparse tactile cues; inspired by Predictive Coding.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作策略 |
| WM | World Model | 视触觉前向预测 |
| WAM | World Action Model | 世界–动作联合建模 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |

## 为什么重要

- Addresses modality collapse where high-bandwidth visual features overshadow sparse tactile cues; inspired by Predictive Coding.
- 想横向对照同一分组的其他工作，可以从 [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Touch](../entities/awesome-touch.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  subgraph PRE["阶段一：CMP 在多任务交互数据上独立预训练，随后冻结"]
    W["腕部相机图像 V_wrist"] --> VE["Vision Encoder<br/>ResNet-18 + MLP 头"]
    VE --> PRIOR["预测先验 ẑ_t (均值 μ_t) + 标量标准差 σ_t"]
    T["触觉图像 I_tac"] --> TE["触觉编码器 (基于 UniT)<br/>→ z_t"]
    PRIOR --> RES["残差触觉 r_t = z_t − ẑ_t"]
    TE --> RES
    RES --> EV["事件编码器 f_ϕ<br/>卷积残差块 + 全局最大池化 → h_t"]
    EV --> VQ["VQ 码本 → Latent Contact Primitive q_t"]
    VQ --> FILM["FiLM 调制先验 ẑ_t → 解码重建触觉图像"]
    PRIOR --> FILM
  end
  subgraph POL["阶段二：整体策略端到端微调（基于 π0.5，CMP 冻结）"]
    SIG["σ_t 作为跨模态 surprise"] --> GATE["Surprise-Aware Gate<br/>g_t = Sigmoid(MLP(σ_t))"]
    VQ --> PROJ["线性层投影 → p_t"]
    GATE --> ET["门控插值<br/>e_t = g_t · p_t + (1−g_t) · e_0（'无接触' token）"]
    PROJ --> ET
    CAM["多相机 RGB + 语言指令 + 本体状态<br/>PaliGemma 编码"] --> ACT
    ET -->|"与噪声 token 拼接"| ACT["动作专家：conditional flow matching 去噪"]
    ACT --> OUT["动作序列 A_t"]
  end
  PRIOR -.-> SIG
```

图依据：论文 §III-A/B/C / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 003/065 |
| 分组 | 21 Unified Vision-Tactile-Language-Action Frameworks |
| 出处 | arXiv 2026 (IROS 2026) |
| 论文 | <https://arxiv.org/abs/2607.03387> |

## 核心机制（归纳）

### 策展导读要点

Addresses modality collapse where high-bandwidth visual features overshadow sparse tactile cues; inspired by Predictive Coding.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.03387) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-touch-technology-map.md) 的 **21 Unified Vision-Tactile-Language-Action Frameworks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Feeling the Unexpected」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **21 Unified Vision-Tactile-Language-Action Frameworks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Touch](../entities/awesome-touch.md)
- 技术地图：[Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_touch_2607_03387_feeling-the-unexpected-restacvla-for-con.md`](../../sources/papers/sun_awesome_touch_2607_03387_feeling-the-unexpected-restacvla-for-con.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_touch_catalog.md`](../../sources/papers/sun_awesome_touch_catalog.md) — 列表总表
- [`sources/repos/awesome-touch.md`](../../sources/repos/awesome-touch.md)
- 论文：<https://arxiv.org/abs/2607.03387>
- [arXiv PDF](https://arxiv.org/pdf/2607.03387)：已通读方法部分（实验与附录以略读为准）：§III Methodology（A–C）、§IV-A 实验设置

## 推荐继续阅读

- [Awesome Touch 仓库](https://github.com/sun254667/awesome-touch)
- [原文](https://arxiv.org/abs/2607.03387)
