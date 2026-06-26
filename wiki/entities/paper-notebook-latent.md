---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, unitree]
status: stub
updated: 2026-06-25
arxiv: "2603.12686"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-notebook-learning-human-like-badminton-skills-for-humanoi.md
sources:
  - ../../sources/papers/humanoid_pnb_latent.md
summary: "LATENT 只用 5 小时、3 × 5 米小场地采集的\"业余网球动作碎片\"，就把 Unitree G1 训练成会在真人对打下完成连续多拍回合的\"人形网球手\"——核心办法是先用动作跟踪器学出一个可修正的 latent 动作空间，再让高层策略在该空间里做 \"修正 + 组合\"，并用 Latent Action Barrier (LAB) 约束策略别跑出先验分布。"
---

# LATENT

**LATENT: Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

LATENT 只用 5 小时、3 × 5 米小场地采集的"业余网球动作碎片"，就把 Unitree G1 训练成会在真人对打下完成连续多拍回合的"人形网球手"——核心办法是先用动作跟踪器学出一个可修正的 latent 动作空间，再让高层策略在该空间里做 "修正 + 组合"，并用 Latent Action Barrier (LAB) 约束策略别跑出先验分布。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat.html> |
| arXiv | <https://arxiv.org/abs/2603.12686> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_latent.md](../../sources/papers/humanoid_pnb_latent.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat.html>
- 论文：<https://arxiv.org/abs/2603.12686>

## 推荐继续阅读

- [机器人论文阅读笔记：LATENT](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat/LATENT__Learning_Athletic_Humanoid_Tennis_Skills_from_Imperfect_Human_Motion_Dat.html)
