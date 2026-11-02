---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2603.10422"
venue: "2026"
summary: "RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2603_10422_world2act-latent-action-post-training-fr.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# World2Act

**World2Act: Latent Action Post-Training from World Model Dynamics** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 552/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

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
    subgraph S1["阶段 1：潜空间对齐（专家演示）"]
        DEMO["专家轨迹视频"] --> WM1["冻结世界模型 Cosmos-Predict2<br/>得到视频 latent V"]
        WM1 --> BV["CNN Video Adapter Bv → zv"]
        ACTS["专家动作序列，按 M 步分块"] --> BA["MLP Action Adapter Ba → za"]
        BA --> DA["MLP Action Decoder Da<br/>重建动作块（Lrecon）"]
        BV --> CON["分块平均余弦相似度的双向 InfoNCE<br/>对齐 zv 与 za（Lcontrastive）"]
        BA --> CON
    end
    subgraph S2["阶段 2：世界模型引导的潜空间残差后训练"]
        OBS["状态 st + 指令 l"] --> VLA["冻结 VLA πbase → 基础动作块"]
        VLA --> BA2["冻结 Ba → za_base"]
        BA2 --> RES["残差策略 fθ（Transformer）预测 Δza"]
        RES --> ZF["za_final = za_base 叠加 Δza"]
        ZF --> DA2["冻结 Da 解码 → 动作块<br/>开环执行 M 步"]
        DA2 --> SIM["B 个并行仿真环境 rollout<br/>仅取初始条件，不用奖励/成功标签"]
        SIM --> WM2["冻结世界模型按初始观测 + 指令生成视频 latent → 冻结 Bv → zv"]
        WM2 --> LOSS["Lcontrastive(zv, za_final)<br/>同一 rollout 为正样本，梯度只更新 fθ"]
        ZF --> LOSS
    end
    S1 -->|"冻结 Bv、Ba、Da"| S2
```

图依据：论文 §3.2–3.3 / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 552/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2603.10422> |
| 项目页 | <https://wm2act.github.io/> |
| 子类 / 象限 | 策略后训练与WM-RL · 泛化与动作对齐 · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（WAMs）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.10422) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「World2Act」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2603_10422_world2act-latent-action-post-training-fr.md`](../../sources/papers/rcl_awesome_wam_2603_10422_world2act-latent-action-post-training-fr.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2603.10422>
- [arXiv PDF](https://arxiv.org/pdf/2603.10422)：已通读方法部分（实验与附录以略读为准）：§3（§3.1 Preliminaries、§3.2 Stage 1、§3.3 Stage 2）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2603.10422)
