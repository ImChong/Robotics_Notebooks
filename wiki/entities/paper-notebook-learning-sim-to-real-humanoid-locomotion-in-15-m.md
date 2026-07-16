---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-16
arxiv: "2512.01996"
related:
  - ../methods/flashsac.md
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-sim-to-real-humanoid-locomotion-in-15-m.md
  - ../../sources/papers/flashsac_arxiv_2604_04539.md
summary: "在 单张 RTX 4090 + 数千并行仿真环境 下，用 为大规模并行调参的 FastSAC / FastTD3（离策略 RL） 配合 极简奖励 + 强域随机化（动力学、粗糙地形、推扰、延迟等），把 全关节人形速度跟踪 的训练墙钟时间压到约 15 分钟，并在 G1 / T1 上完成 sim-to-real；同一套配方也可加速 全身人形动作跟踪（相对 PPO 更快）。"
---

# Learning Sim-to-Real Humanoid Locomotion in 15 Minutes

**Learning Sim-to-Real Humanoid Locomotion in 15 Minutes** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

在 单张 RTX 4090 + 数千并行仿真环境 下，用 为大规模并行调参的 FastSAC / FastTD3（离策略 RL） 配合 极简奖励 + 强域随机化（动力学、粗糙地形、推扰、延迟等），把 全关节人形速度跟踪 的训练墙钟时间压到约 15 分钟，并在 G1 / T1 上完成 sim-to-real；同一套配方也可加速 全身人形动作跟踪（相对 PPO 更快）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。
- 后续 **[FlashSAC](../methods/flashsac.md)**（arXiv:2604.04539）在同一 off-policy 墙钟加速脉络上，用更大网络与范数约束进一步挑战 PPO 默认地位；本文 FastSAC/FastTD3 配方可视为其前驱小网络路线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 03_High_Impact_Selection |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes.html> |
| arXiv | <https://arxiv.org/abs/2512.01996> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- [FlashSAC（方法页）](../methods/flashsac.md)
- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-sim-to-real-humanoid-locomotion-in-15-m.md](../../sources/papers/humanoid_pnb_learning-sim-to-real-humanoid-locomotion-in-15-m.md)
- [flashsac_arxiv_2604_04539.md](../../sources/papers/flashsac_arxiv_2604_04539.md) — 后继 scaling 式 off-policy（FlashSAC）
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes.html>
- 论文：<https://arxiv.org/abs/2512.01996>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes/Learning_Sim-to-Real_Humanoid_Locomotion_in_15_Minutes.html)
