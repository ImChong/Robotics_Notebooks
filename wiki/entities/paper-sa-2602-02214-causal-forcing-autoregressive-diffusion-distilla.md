---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.02214"
venue: "arXiv 2026"
code: https://github.com/thu-ml/Causal-Forcing
summary: "Real-time interactive video generation via autoregressive diffusion distillation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_02214_causal-forcing-autoregressive-diffusion.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Causal Forcing

**Causal Forcing: Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 073/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Real-time interactive video generation via autoregressive diffusion distillation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Real-time interactive video generation via autoregressive diffusion distillation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    B["预训练双向视频扩散模型"] -.->|"现有做法（Self Forcing）：<br/>直接蒸馏 AR 学生，违反 frame-level injectivity"| X["ODE 初始化效果差（条件期望解，视频模糊）"]
    D["真实视频数据"] --> S1["阶段1：自回归扩散训练<br/>采用 Teacher Forcing（以干净前缀为条件）<br/>优于 Diffusion Forcing"]
    S1 --> T["AR 扩散模型 = 教师"]
    T --> S2A["采样教师 PF-ODE 轨迹<br/>真实干净帧作历史，从高斯噪声生成当前帧"]
    D --> S2A
    S2A --> S2["阶段2：因果 ODE 蒸馏<br/>学生 G_θ 由 (x_t^i, 干净历史) 回归干净帧 x_0^i"]
    S2 --> S3["阶段3：非对称 DMD（同 Self Forcing）<br/>学生以自身生成的前缀自回归 rollout"]
    S3 --> OUT["少步自回归学生模型<br/>实时交互视频生成"]
    S2 -.->|"扩展（§3.4）"| CD["因果一致性蒸馏 Causal CD<br/>可替代阶段2，留作未来工作"]
```

图依据：论文 §3.2-3.4（现有方法分析与 Causal Forcing 三阶段流程）、Fig.3、Fig.5。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 073/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2602.02214> |
| 代码/项目 | <https://github.com/thu-ml/Causal-Forcing> |

## 核心机制（归纳）

### 策展导读要点

Real-time interactive video generation via autoregressive diffusion distillation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.02214) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Causal Forcing」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_02214_causal-forcing-autoregressive-diffusion.md`](../../sources/papers/sun_awesome_wm_2602_02214_causal-forcing-autoregressive-diffusion.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.02214>
- [arXiv PDF](https://arxiv.org/pdf/2602.02214)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 Background、§3 Method（3.1-3.4）。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.02214)
