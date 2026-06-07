---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2601.22517"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robostriker.md
summary: "RoboStriker 把\"两个人形机器人互殴\"建成 两玩家零和马尔可夫博弈，先用单智能体追真人拳击 MoCap 训出 运动跟踪器（46 段、约 14 分钟 Xsens 数据，经 GMR 重定向到 Unitree G1），再把这些技能蒸馏成一个 投到单位超球面的潜空间动作流形，最后在这个潜空间上跑 Latent-Space Neural Fictitious Self-Play (LS-NFSP)，让两个智能体只挑\"高层动作意图\"而不直接挑电机指令——动作天然物理可行又像人，多智能体训练也稳定收敛。"
---

# RoboStriker

**RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

RoboStriker 把"两个人形机器人互殴"建成 两玩家零和马尔可夫博弈，先用单智能体追真人拳击 MoCap 训出 运动跟踪器（46 段、约 14 分钟 Xsens 数据，经 GMR 重定向到 Unitree G1），再把这些技能蒸馏成一个 投到单位超球面的潜空间动作流形，最后在这个潜空间上跑 Latent-Space Neural Fictitious Self-Play (LS-NFSP)，让两个智能体只挑"高层动作意图"而不直接挑电机指令——动作天然物理可行又像人，多智能体训练也稳定收敛。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing.html> |
| arXiv | <https://arxiv.org/abs/2601.22517> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robostriker.md](../../sources/papers/humanoid_pnb_robostriker.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing.html>
- 论文：<https://arxiv.org/abs/2601.22517>

## 推荐继续阅读

- [机器人论文阅读笔记：RoboStriker](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing/RoboStriker__Hierarchical_Decision-Making_for_Autonomous_Humanoid_Boxing.html)
