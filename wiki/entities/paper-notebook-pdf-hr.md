---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2602.04851"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_pdf-hr.md
summary: "PDF-HR 训练一个神经距离场：输入一个机器人姿态，输出它到\"已知合理姿态语料库\"的最短距离；这个连续可微的\"姿态可信度\"可以直接当作 RL 的奖励整形项、模仿学习的正则项，或作为运动重定向时的姿态打分器，几乎零成本接入到现有 humanoid 任务流水线。"
---

# PDF-HR

**PDF-HR: Pose Distance Fields for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

PDF-HR 训练一个神经距离场：输入一个机器人姿态，输出它到"已知合理姿态语料库"的最短距离；这个连续可微的"姿态可信度"可以直接当作 RL 的奖励整形项、模仿学习的正则项，或作为运动重定向时的姿态打分器，几乎零成本接入到现有 humanoid 任务流水线。

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
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2602.04851> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_pdf-hr.md](../../sources/papers/humanoid_pnb_pdf-hr.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2602.04851>

## 推荐继续阅读

- [机器人论文阅读笔记：PDF-HR](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots/PDF-HR__Pose_Distance_Fields_for_Humanoid_Robots.html)
