---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2606.12690"
summary: "In this paper, we propose the Enhanced World Action Model (EWAM), a closed-loop online adaptation architecture built upon a pretrained and fully frozen Cosmos3 backbone network."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2606_12690_ewam-an-enhanced-world-action-model-for.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# EWAM

**EWAM: An Enhanced World Action Model for Closed-Loop Online Adaptation in Embodied Intelligence** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 336/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

In this paper, we propose the Enhanced World Action Model (EWAM), a closed-loop online adaptation architecture built upon a pretrained and fully frozen Cosmos3 backbone network.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- In this paper, we propose the Enhanced World Action Model (EWAM), a closed-loop online adaptation architecture built upon a pretrained and fully frozen Cosmos3 backbone network.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["视觉观测 o_t、本体状态 q_t、语言指令 l"] --> VLE["视觉-语言编码器 VLE（冻结）→ h_VLE"]
    VLE --> AR["自回归 Reasoner（冻结）→ 推理上下文 c_t"]
    AR --> DIT["DiT 动作生成器（冻结），其第 l_mem = 6 层插入<br/>神经经验记忆层"]
    MEM[("经验记忆 M<br/>top-k = 5 检索")] --> DIT
    DIT --> SH["状态预测头 → ŝ(t+1)"]
    DIT --> AH["动作头 → 原始动作块 a0"]
    SH --> AN["神经异常检测层 → 异常向量 ι_t<br/>（预测残差、动力学不一致、碰撞、空抓、幻觉、力违规）"]
    AH --> AN
    AN --> RT["神经策略路由层 → r_t"]
    RT -->|"direct"| COR["神经动作修正层 → 修正动作 a*"]
    RT -->|"conservative"| COR
    RT -->|"rollback：从记忆取最近稳定状态的保守动作"| COR
    AH --> COR
    COR --> ENV["环境执行"]
    ENV --> FLT["经验过滤质量门（安全 / 任务 / 效率阈值）"]
    FLT -->|"合格"| UPD["写入记忆 M；间歇更新四个神经层与轻量 adapter"]
    UPD --> MEM
    FLT -->|"不合格"| LOG["仅记录诊断，不入记忆、不参与更新"]
```

图依据：论文 §3.1–3.10 / Fig.1、Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 336/564 |
| 分组 | WAMs |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2606.12690> |
| 子类 / 象限 | 记忆与长时序 · 策略后训练与WM-RL · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

In this paper, we propose the Enhanced World Action Model (EWAM), a closed-loop online adaptation architecture built upon a pretrained and fully frozen Cosmos3 backbone network.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.12690) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EWAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2606_12690_ewam-an-enhanced-world-action-model-for.md`](../../sources/papers/rcl_awesome_wam_2606_12690_ewam-an-enhanced-world-action-model-for.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2606.12690>
- [arXiv PDF](https://arxiv.org/pdf/2606.12690)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1–3.11；实验 §4 略读）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2606.12690)
