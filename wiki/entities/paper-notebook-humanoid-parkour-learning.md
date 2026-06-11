---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-11
arxiv: "2406.10759"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-parkour-learning.md
summary: "用 RL 在仿真里练「神谕」地形策略（scandots 地形编码 + GRU 状态估计），再通过 DAgger 把感知蒸馏成机载深度图 CNN，在 无参考轨迹、无抬脚奖励项 的前提下，让人形在多种跑酷障碍上 零样本 sim-to-real，并能跟随摇杆转向命令；手臂通道可覆盖以迁移到移动操作。"
---

# Humanoid Parkour Learning

**Humanoid Parkour Learning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

用 RL 在仿真里练「神谕」地形策略（scandots 地形编码 + GRU 状态估计），再通过 DAgger 把感知蒸馏成机载深度图 CNN，在 无参考轨迹、无抬脚奖励项 的前提下，让人形在多种跑酷障碍上 零样本 sim-to-real，并能跟随摇杆转向命令；手臂通道可覆盖以迁移到移动操作。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Parkour_Learning/Humanoid_Parkour_Learning.html> |
| arXiv | <https://arxiv.org/abs/2406.10759> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-parkour-learning.md](../../sources/papers/humanoid_pnb_humanoid-parkour-learning.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Parkour_Learning/Humanoid_Parkour_Learning.html>
- 论文：<https://arxiv.org/abs/2406.10759>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid Parkour Learning](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Humanoid_Parkour_Learning/Humanoid_Parkour_Learning.html)
