---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2602.08518"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_characteristics-management-and-utilization-of-mu.md
summary: "把 JSK 实验室十几年在 腱-驱动肌骨型人形 上的「设计 / 控制 / 学习」经验，浓缩成一篇「特性 → 管理 → 利用」的三段式实证综述：先把肌肉的五大固有特性（冗余 / 独立 / 各向异性 / 可变力臂 / 非线性弹性）讲清楚，再讲怎么用硬件模块把这些特性\"管住\"，最后讲怎么用反射 + 学习的方法把它们\"用好\"。"
---

# Characteristics, Management, and Utilization of Muscles in Musculoskeletal Humanoids

**Characteristics, Management, and Utilization of Muscles in Musculoskeletal Humanoids** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把 JSK 实验室十几年在 腱-驱动肌骨型人形 上的「设计 / 控制 / 学习」经验，浓缩成一篇「特性 → 管理 → 利用」的三段式实证综述：先把肌肉的五大固有特性（冗余 / 独立 / 各向异性 / 可变力臂 / 非线性弹性）讲清楚，再讲怎么用硬件模块把这些特性"管住"，最后讲怎么用反射 + 学习的方法把它们"用好"。

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
| 分类 | 12_Hardware_Design |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids.html> |
| arXiv | <https://arxiv.org/abs/2602.08518> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_characteristics-management-and-utilization-of-mu.md](../../sources/papers/humanoid_pnb_characteristics-management-and-utilization-of-mu.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids.html>
- 论文：<https://arxiv.org/abs/2602.08518>

## 推荐继续阅读

- [机器人论文阅读笔记：Characteristics, Management, and Utilization of Muscles in Musculoskeletal Humanoids](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids/Characteristics_Management_and_Utilization_of_Muscles_in_Musculoskeletal_Humanoids.html)
