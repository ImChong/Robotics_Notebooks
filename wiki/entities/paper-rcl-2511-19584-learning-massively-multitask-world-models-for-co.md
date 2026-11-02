---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2511.19584"
venue: "2025"
summary: "Newt extends TD-MPC2 into a language-conditioned agent that learns latent dynamics from demonstrations and online interaction across MMBench. Its practical recipe pretrains the world model and policy prior, retains actio"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2511_19584_learning-massively-multitask-world-model.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Learning Massively Multitask World Models for Continuous Control

**Learning Massively Multitask World Models for Continuous Control** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 410/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Newt extends TD-MPC2 into a language-conditioned agent that learns latent dynamics from demonstrations and online interaction across MMBench. Its practical recipe pretrains the world model and policy prior, retains action supervision, and uses the model to plan. It improves over the evaluated multitask baselines, bu...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- Newt extends TD-MPC2 into a language-conditioned agent that learns latent dynamics from demonstrations and online interaction across MMBench. Its practical recipe pretrains the world model and policy prior, retains action supervision, and uses the model to plan. It improves over the evaluated multitask baselines, bu...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    DEMO["每任务 10–40 条演示<br/>(单任务 TD-MPC2 智能体采集)"] --> PRE["模型预训练：在演示上联合优化<br/>L(θ) + Lp(θ)，Q 值项暂时关闭"]
    PRE --> WM
    subgraph WM["Newt 世界模型（self-predictive，无解码器）"]
        LANG["语言指令 → 冻结 CLIP-text 编码 g"] --> ENC
        IMG["RGB 观测（可选）→ 冻结 DINOv2 编码 x"] --> ENC
        STATE["状态向量 s_state（带 valid mask）"] --> ENC
        ENC["状态编码器 h → 潜状态 z"] --> DYN["潜在动力学 d(z,a,g)"]
        DYN --> RQ["奖励 R / 价值 Q 预测<br/>(离散回归 CE)"]
        ENC --> PI["策略先验 p(z,g)"]
    end
    WM --> PLAN["规划：以策略先验热启动<br/>初期偏向预训练策略，约前 12% 训练线性退火至 0"]
    PLAN --> ENV["多任务环境交互<br/>200 个任务 / 10 个域"]
    ENV --> BUF_ON["在线交互回放缓冲"]
    DEMO --> BUF_DEMO["演示回放缓冲"]
    BUF_ON -->|"采样 50%"| UPD
    BUF_DEMO -->|"采样 50%"| UPD
    UPD["联合更新：L(θ) 自预测+奖励+价值<br/>Lp(θ) = model-based BC + Q 值 + 熵"] --> WM
```

图依据：论文 §3.1 式 (1)–(3) 与 §3.2 / Fig. 6（方法）；MMBench 基准本身（§2）不在图中展开。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 410/564 |
| 分组 | WAMs |
| 出处 | 2025 |
| 论文 | <https://arxiv.org/abs/2511.19584> |
| 子类 / 象限 | 策略后训练与WM-RL · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

Newt extends TD-MPC2 into a language-conditioned agent that learns latent dynamics from demonstrations and online interaction across MMBench. Its practical recipe pretrains the world model and policy prior, retains action supervision, and uses the model to plan. It improves over the evaluated multitask baselines, bu...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2511.19584) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Learning Massively Multitask World Models for Continuous Control」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2511_19584_learning-massively-multitask-world-model.md`](../../sources/papers/rcl_awesome_wam_2511_19584_learning-massively-multitask-world-model.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2511.19584>
- [arXiv PDF](https://arxiv.org/pdf/2511.19584)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 基准、§3 Newt 方法（3.1–3.2）、§4 实验概览

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2511.19584)
