---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2502.01143"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_asap-aligning-simulation-and-real-world-physics.md
summary: "先用人类视频重定向后的参考动作在仿真里预训练运动跟踪策略，再在真机 rollout 收集状态轨迹，用残差（delta）动作模型显式补偿仿真与真机的动力学差；把该模型冻结后嵌入仿真器做「物理对齐」式的策略微调，最后在真机去掉 delta 模型直接部署——在侧跳、前跳、踢球、球星庆祝动作等全身敏捷技能上显著降低跟踪误差，并优于纯 SysID、纯域随机化、以及仅学习 delta 动力学但不回灌仿真的基线。"
---

# ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Skills

**ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Skills** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

先用人类视频重定向后的参考动作在仿真里预训练运动跟踪策略，再在真机 rollout 收集状态轨迹，用残差（delta）动作模型显式补偿仿真与真机的动力学差；把该模型冻结后嵌入仿真器做「物理对齐」式的策略微调，最后在真机去掉 delta 模型直接部署——在侧跳、前跳、踢球、球星庆祝动作等全身敏捷技能上显著降低跟踪误差，并优于纯 SysID、纯域随机化、以及仅学习 delta 动力学但不回灌仿真的基线。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills.html> |
| arXiv | <https://arxiv.org/abs/2502.01143> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_asap-aligning-simulation-and-real-world-physics.md](../../sources/papers/humanoid_pnb_asap-aligning-simulation-and-real-world-physics.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills.html>
- 论文：<https://arxiv.org/abs/2502.01143>

## 推荐继续阅读

- [机器人论文阅读笔记：ASAP Aligning Simulation and Real-World Physics for Agile Humanoid Skills](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills/ASAP_Aligning_Simulation_and_Real-World_Physics_for_Agile_Humanoid_Skills.html)
