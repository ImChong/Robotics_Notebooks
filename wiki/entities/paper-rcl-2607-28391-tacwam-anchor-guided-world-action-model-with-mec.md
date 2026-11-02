---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2607.28391"
summary: "We present TacWAM, a mechanics-aware tactile WAM that addresses this challenge in three steps. TacWAM achieves an average success rate of 75.0%, exceeding the strongest evaluated baseline by 37.5 percentage points."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2607_28391_tacwam-anchor-guided-world-action-model.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# TacWAM

**TacWAM: Anchor-Guided World Action Model with Mechanics-Aware Tactile Prediction** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 490/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We present TacWAM, a mechanics-aware tactile WAM that addresses this challenge in three steps. TacWAM achieves an average success rate of 75.0%, exceeding the strongest evaluated baseline by 37.5 percentage points.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We present TacWAM, a mechanics-aware tactile WAM that addresses this challenge in three steps. TacWAM achieves an average success rate of 75.0%, exceeding the strongest evaluated baseline by 37.5 percentage points.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    OBS["观测：视觉 o_v、触觉 o_tac（修正图像 + 稠密力场 + 网格形变流）、本体状态 q、任务语言 l"] --> SAF["阶段一：SAF 触觉编码器 E_SAF 得到触觉 latent z_tac；R_tac 重建力场、形变流与双侧合力矩（wrench）"]
    SAF --> FREEZE["冻结触觉编码器与解码器"]
    FREEZE --> HIST["触觉历史编码器 E_hist：近期 z_tac → 块级触觉上下文 c_tac（经自适应归一化调制触觉专家）"]
    FREEZE --> TOK["token 排布：[V0, V1:H | T0, T1:H | A]（V0、T0 为干净锚点）"]
    OBS --> TOK
    TOK --> MOT["三流 Mixture-of-Transformers：视觉 / 触觉 / 动作专家，AGT 掩码约束的混合自注意力；语言与本体经 cross-attention 提供"]
    HIST --> MOT
    MOT --> VF["未来视觉预测 V1:H（仅训练监督）"]
    MOT --> TF["未来触觉预测 T1:H；冻结可微 R_tac 解码做一致性正则"]
    MOT --> ACT["动作块 A：仅读 V0、T0 与动作 token，不能读未来视觉/触觉 token"]
    ACT --> EXEC["部署：块级推理 → 执行一个动作块 → 新观测 → 下一块（滚动时域闭环）"]
```

图依据：论文 §3.1–§3.6 与 Fig. 1、Table 1（AGT visibility）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 490/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2607.28391> |
| 子类 / 象限 | 多模态触觉音频 · 高效推理与实时控制 · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

We present TacWAM, a mechanics-aware tactile WAM that addresses this challenge in three steps. TacWAM achieves an average success rate of 75.0%, exceeding the strongest evaluated baseline by 37.5 percentage points.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.28391) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TacWAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2607_28391_tacwam-anchor-guided-world-action-model.md`](../../sources/papers/rcl_awesome_wam_2607_28391_tacwam-anchor-guided-world-action-model.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2607.28391>
- [arXiv PDF](https://arxiv.org/pdf/2607.28391)：已通读方法部分（实验与附录以略读为准）：§3 Method（§3.1–§3.6）、§4.1 实验设置

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2607.28391)
