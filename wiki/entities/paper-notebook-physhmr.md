---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2510.02566"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_physhmr.md
summary: "把 HMR 从\"先估姿态、再做物理后修\"的两段式拼接，压缩成一个端到端的视觉条件 RL 策略：用 GVHMR 抽到的视觉特征做局部 pose 推理，加上把 2D 关键点抬成\"3D 射线\"的 pixel-as-ray 软全局对齐，再叠一层从 MoCap 专家蒸馏来的运动先验 + 物理 reward 微调；输出直接是物理仿真中跑得动、又对齐视频的人体动作。"
---

# PhysHMR

**PhysHMR: Learning Humanoid Control Policies from Vision for Physically Plausible Human Motion Reconstruction** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：13_Physics-Based_Animation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把 HMR 从"先估姿态、再做物理后修"的两段式拼接，压缩成一个端到端的视觉条件 RL 策略：用 GVHMR 抽到的视觉特征做局部 pose 推理，加上把 2D 关键点抬成"3D 射线"的 pixel-as-ray 软全局对齐，再叠一层从 MoCap 专家蒸馏来的运动先验 + 物理 reward 微调；输出直接是物理仿真中跑得动、又对齐视频的人体动作。

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
| 分类 | 13_Physics-Based_Animation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR.html> |
| arXiv | <https://arxiv.org/abs/2510.02566> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_physhmr.md](../../sources/papers/humanoid_pnb_physhmr.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR.html>
- 论文：<https://arxiv.org/abs/2510.02566>

## 推荐继续阅读

- [机器人论文阅读笔记：PhysHMR](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR/PhysHMR__Learning_Humanoid_Control_Policies_from_Vision_for_Physical_HMR.html)
