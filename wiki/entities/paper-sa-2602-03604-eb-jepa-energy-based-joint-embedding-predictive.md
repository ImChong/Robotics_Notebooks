---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.03604"
venue: "arXiv 2026"
code: https://github.com/facebookresearch/eb_jepa
summary: "Energy-based joint-embedding predictive architectures."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_03604_eb-jepa-energy-based-joint-embedding-pre.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EB-JEPA

**EB-JEPA: Energy-Based Joint-Embedding Predictive Architectures** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 112/571** 篇，分组 **44 Latent Space World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Energy-based joint-embedding predictive architectures.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Energy-based joint-embedding predictive architectures.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph IMG["(a) Image-JEPA"]
        X["图像 x"] --> V["随机裁剪 / 颜色抖动<br/>得到两个视图 x, x'"]
        V --> E1["编码器 f_θ → z, z'"]
        E1 --> L1["视图不变损失 ‖z−z'‖² + λR"]
    end
    subgraph VID["(b) Video-JEPA"]
        XT["视频帧 x_1:T"] --> E2["编码器 f_θ → z_t"]
        E2 --> P2["预测器 g_φ<br/>由 z_t-v:t 预测 z_t+1"]
        P2 --> L2["Lpred + λR<br/>（k 步递归 rollout 损失）"]
    end
    subgraph AC["(c) 动作条件 Video-JEPA（世界模型）"]
        XA["观测 x_t"] --> E3["编码器 f_θ → z_t"]
        AT["动作 a_t"] --> Q["动作编码器 q_ω → u_t"]
        E3 --> P3["预测器 g_φ(z, u) → z_t+1"]
        Q --> P3
        P3 --> L3["Lpred + VICReg 项 + Lsim + L_IDM"]
    end
    R["防塌缩正则 R：VICReg 或 SIGReg<br/>（作用于投影空间）"] -.-> L1
    R -.-> L2
    R -.-> L3
    L3 --> W["训练好的潜空间动力学模型"]
    W --> PL["(d) 目标条件规划<br/>MPPI 优化动作序列 a_0:H"]
    G["目标观测 x_g → f_θ → z_g"] --> PL
    PL --> OUT["在潜空间 rollout，最小化到 z_g 的累计距离<br/>（Eplan）"]
```

图依据：论文 §3（统一 JEPA 框架，式 1-5、13）、§4（多步 rollout 训练与目标条件规划，式 9-14）、Fig.1。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 112/571 |
| 分组 | 44 Latent Space World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.03604> |
| 代码/项目 | <https://github.com/facebookresearch/eb_jepa> |

## 核心机制（归纳）

### 策展导读要点

Energy-based joint-embedding predictive architectures.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.03604) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **44 Latent Space World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EB-JEPA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **44 Latent Space World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_03604_eb-jepa-energy-based-joint-embedding-pre.md`](../../sources/papers/sun_awesome_wm_2602_03604_eb-jepa-energy-based-joint-embedding-pre.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.03604>
- [arXiv PDF](https://arxiv.org/pdf/2602.03604)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 Preliminaries、§4 Training and Planning with World Models、§5 实验设定。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.03604)
