---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2602.06445"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_eco-energy-constrained-optimization-with-rl-for.md
summary: "把电机能耗从「多目标奖励里的一堆加权项」里拆出来，改成 CMDP 下的显式不等式约束（再配合镜像对称 / 参考运动类约束），用 PPO-Lagrangian 在仿真里稳定求解，并在 BRUCE 上实现比 MPC、普通 PPO 显著更低能耗的稳健对称行走。"
---

# ECO Energy Constrained Optimization with RL for Humanoid Walking

**ECO Energy Constrained Optimization with RL for Humanoid Walking** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把电机能耗从「多目标奖励里的一堆加权项」里拆出来，改成 CMDP 下的显式不等式约束（再配合镜像对称 / 参考运动类约束），用 PPO-Lagrangian 在仿真里稳定求解，并在 BRUCE 上实现比 MPC、普通 PPO 显著更低能耗的稳健对称行走。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 03_High_Impact_Selection |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking.html> |
| arXiv | <https://arxiv.org/abs/2602.06445> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_eco-energy-constrained-optimization-with-rl-for.md](../../sources/papers/humanoid_pnb_eco-energy-constrained-optimization-with-rl-for.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking.html>
- 论文：<https://arxiv.org/abs/2602.06445>

## 推荐继续阅读

- [机器人论文阅读笔记：ECO Energy Constrained Optimization with RL for Humanoid Walking](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking/ECO_Energy_Constrained_Optimization_with_RL_for_Humanoid_Walking.html)
