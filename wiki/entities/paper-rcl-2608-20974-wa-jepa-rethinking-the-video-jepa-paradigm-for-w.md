---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2608.20974"
code: https://github.com/AFARI-Research/WA-JEPA
summary: "Video Joint Embedding Predictive Architecture (V-JEPA) learns powerful spatiotemporal representations from video through self-supervised latent feature prediction. However, V-JEPA is built around random-mask completion a"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2608_20974_wa-jepa-rethinking-the-video-jepa-paradi.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# WA-JEPA

**WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling in Autonomous Driving** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 530/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Video Joint Embedding Predictive Architecture (V-JEPA) learns powerful spatiotemporal representations from video through self-supervised latent feature prediction. However, V-JEPA is built around random-mask completion and deterministic regression, making it fundamentally ill-suited for autonomous driving planning t...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- Video Joint Embedding Predictive Architecture (V-JEPA) learns powerful spatiotemporal representations from video through self-supervised latent feature prediction. However, V-JEPA is built around random-mask completion and deterministic regression, making it fundamentally ill-suited for autonomous driving planning t...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["多视角历史帧 X1:H<br/>左/前/右/后相机"] --> E["共享在线编码器 Eθ<br/>V-JEPA 2 ViT-L"]
    E --> C["历史上下文 token Zctx"]
    subgraph S1["Stage 1：混合未来掩码预训练（nuPlan，无动作监督）"]
        M["未来帧掩码<br/>Full-mask + Patch-mask"] --> Z["未来条件 token Zcond<br/>可学习 mask token + 可见 token"]
        FF["未来帧 XH+1:H+K"] --> T["EMA 目标编码器"]
        T --> ZS["干净未来目标 Z*future"]
        N1["高斯噪声 + 流时间 t"] --> ZT["带噪未来 token Zt"]
        C --> P1["未来 Flow 预测器<br/>MMDiT 风格"]
        Z --> P1
        ZT --> P1
        P1 --> L1["MSE：预测干净未来 latent vs Z*future"]
        ZS --> L1
    end
    subgraph S2["Stage 2：联合世界-动作建模（NAVSIM 微调，仅 Full-mask）"]
        AH["历史动作 Y1:H + 自车状态 s"] --> AT["动作 token Tact"]
        NA["带噪未来动作"] --> AT
        C --> JP["联合未来-动作预测器<br/>MMDiT 风格"]
        AT --> JP
        JP --> FS["未来场景流"]
        JP --> AS["动作流"]
        AS --> LA["MSE：归一化未来动作"]
    end
    S1 -->|"初始化权重"| S2
    AS --> OUT["推理：噪声迭代采样 4 步 → 反归一化 → 未来自车轨迹"]
```

图依据：论文 §Method（Overview、Stage 1、Stage 2、Planning inference）/ Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 530/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2608.20974> |
| 代码/项目 | <https://github.com/AFARI-Research/WA-JEPA> |
| 子类 / 象限 | 自动驾驶 · 潜空间预测与JEPA · 联合视频动作建模 · Q3 · Dual-system × 联合预测 |

## 核心机制（归纳）

### 策展导读要点

Video Joint Embedding Predictive Architecture (V-JEPA) learns powerful spatiotemporal representations from video through self-supervised latent feature prediction. However, V-JEPA is built around random-mask completion and deterministic regression, making it fundamentally ill-suited for autonomous driving planning t...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2608.20974) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WA-JEPA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2608_20974_wa-jepa-rethinking-the-video-jepa-paradi.md`](../../sources/papers/rcl_awesome_wam_2608_20974_wa-jepa-rethinking-the-video-jepa-paradi.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2608.20974>
- [arXiv PDF](https://arxiv.org/pdf/2608.20974)：已通读方法部分（实验与附录以略读为准）：Method（Preliminaries、Overview、Stage 1、Stage 2、Planning inference）与 Experimental Setup 实现细节

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2608.20974)
