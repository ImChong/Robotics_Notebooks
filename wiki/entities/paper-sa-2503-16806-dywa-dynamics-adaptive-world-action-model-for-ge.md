---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm, awesome-world-action-models-rcl]
status: complete
updated: 2026-10-06
arxiv: "2503.16806"
venue: "arXiv 2025"
code: https://github.com/jiangranlv/DyWA
summary: "Dynamics-adaptive world action model for generalizable non-prehensile manipulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_16806_dywa-dynamics-adaptive-world-action-mode.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
  - ../../sources/papers/rcl_awesome_wam_ref_283b7da95c4145cf56d0_dywa-dynamics-adaptive-world-action-mode.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# DyWA

**DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 288/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Dynamics-adaptive world action model for generalizable non-prehensile manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Dynamics-adaptive world action model for generalizable non-prehensile manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  TE["教师策略：PPO 训练 200K 迭代，特权信息为完整物体点云、任务状态、物理参数"] --> DA["DAgger 蒸馏 500K 迭代：模仿损失 + 适应损失监督学生"]
  OBS["学生观测：局部点云 P_t、关节状态 J_t、末端位姿 E_t"] --> ENC["PointNet++ 编码点云；浅层 MLP 编码关节与末端位姿"]
  GOAL["目标：初始点云按目标位姿变换得 P_G，共用点云编码器"] --> ENC
  HIST["最近 L 步观测嵌入与上一步动作嵌入"] --> ADA["1D CNN 适应模块 → 适应嵌入 z_t"]
  ADA --> DYN["解码为动力学嵌入 → FiLM 的 γ、β，调制世界动作模型前部层"]
  ENC --> WAM["世界动作模型（MLP）"]
  DYN --> WAM
  WAM --> ACT["动作 A_t：末端子目标残差 ΔT_ee + 关节阻抗参数 P、ρ"]
  WAM --> NS["预测下一任务状态 S_t+1（平移 T、旋转 R），世界模型损失"]
  ACT --> IK["阻尼最小二乘 IK 得 q_d → 关节空间阻抗控制器"]
  DA -.-> WAM
  DA -.-> ADA
```

图依据：论文 §3.1–§3.6、附录 §7（教师策略）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 288/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.16806> |
| 代码/项目 | <https://github.com/jiangranlv/DyWA> |

## 核心机制（归纳）

### 策展导读要点

Dynamics-adaptive world action model for generalizable non-prehensile manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.16806) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DyWA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2503_16806_dywa-dynamics-adaptive-world-action-mode.md`](../../sources/papers/sun_awesome_wm_2503_16806_dywa-dynamics-adaptive-world-action-mode.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.16806>
- [`sources/papers/rcl_awesome_wam_ref_283b7da95c4145cf56d0_dywa-dynamics-adaptive-world-action-mode.md`](../../sources/papers/rcl_awesome_wam_ref_283b7da95c4145cf56d0_dywa-dynamics-adaptive-world-action-mode.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- [arXiv PDF](https://arxiv.org/pdf/2503.16806)：已通读方法部分（实验与附录以略读为准）：§3（3.1–3.6）、附录 §7（RL 教师策略）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.16806)
