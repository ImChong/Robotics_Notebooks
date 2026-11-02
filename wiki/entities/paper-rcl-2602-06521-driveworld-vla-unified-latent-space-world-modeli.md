---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2602.06521"
venue: "ICML 2026"
code: https://github.com/liulin815/DriveWorld-VLA
summary: "RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2602_06521_driveworld-vla-unified-latent-space-worl.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# DriveWorld-VLA

**DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 326/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  IN["多视角图像 I_t + 文本提示 T_t + 历史动作 A_t-1（转为文本） + BEV 特征 B_t（BEVFormer）"] --> TOK["各模态独立 tokenize（BEV tokenizer / 文本 tokenizer）"]
  TOK --> VLM["VLM（InternVL）→ 共享隐表示 H_t"]
  VLM --> XA["Cross-Attention 得 B_t'"]
  subgraph S1["Stage 1 协同训练：未来想象 + 动作预测"]
    XA --> D1["Denoiser 分支 1（历史条件）→ 未来 BEV"]
    VLM --> ACT["轻量 Action Decoder → 未来动作 A'"]
    D1 --> SEG["Seg Head 解码语义 BEV（L_seg）"]
    ACT --> LA["模仿学习（L_act）"]
  end
  subgraph S2["Stage 2 动作可控性微调"]
    XA --> D2["Denoiser 分支 2（DiT，动作条件流匹配）"]
    GTA["真值未来动作"] --> D2
    D2 --> LFM["L_FM"]
  end
  subgraph S3["Stage 3 未来引导评估与修正"]
    ACT --> D2b["预测动作 → 分支 2 Euler 采样（25 步）→ 动作条件未来 BEV"]
    D2b --> RW["Reward 函数 R 打分 r̂"]
    RW --> RA["按 r̂ 加权的动作损失 L_act'；联合 L_seg、L_rew"]
  end
  S1 --> S2 --> S3
```

/tmp/x_31/4.b

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 326/564 |
| 分组 | WAMs |
| 出处 | ICML 2026 |
| 论文 | <https://icml.cc/virtual/2026/poster/61191> |
| 代码/项目 | <https://github.com/liulin815/DriveWorld-VLA> |
| 子类 / 象限 | 自动驾驶 · 潜空间预测与JEPA · 策略后训练与WM-RL · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://icml.cc/virtual/2026/poster/61191) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DriveWorld-VLA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2602_06521_driveworld-vla-unified-latent-space-worl.md`](../../sources/papers/rcl_awesome_wam_2602_06521_driveworld-vla-unified-latent-space-worl.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://icml.cc/virtual/2026/poster/61191>
- [arXiv PDF](https://arxiv.org/pdf/2602.06521)：已通读方法部分（实验与附录以略读为准）：摘要、§3 方法（3.1–3.3，含 Fig. 2–3）、§4.1 实现细节

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://icml.cc/virtual/2026/poster/61191)
