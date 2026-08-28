---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2603.08619"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_embedding-classical-balance-control-principles-i.md
summary: "这篇论文的核心不是“再堆一个更大的网络”，而是把经典平衡控制里可解释的物理量（Capture Point、CoM 状态、质心动量）嵌入到 RL 训练中： - 在训练时作为特权 critic 输入和奖励塑形信号； - 在部署时 actor 仍然只依赖本体感觉，保证可落地。 结果是在一个统一策略中实现了从小扰动到大跌倒后的恢复行为链，并报告 93.4% 的恢复成功率。"
---

# Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery

**Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

这篇论文的核心不是“再堆一个更大的网络”，而是把经典平衡控制里可解释的物理量（Capture Point、CoM 状态、质心动量）嵌入到 RL 训练中： - 在训练时作为特权 critic 输入和奖励塑形信号； - 在部署时 actor 仍然只依赖本体感觉，保证可落地。 结果是在一个统一策略中实现了从小扰动到大跌倒后的恢复行为链，并报告 93.4% 的恢复成功率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery.html> |
| arXiv | <https://arxiv.org/abs/2603.08619> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这篇工作的核心取舍是「经典平衡控制的物理量只进训练、不进部署」：Capture Point、CoM 状态与质心动量用来塑形学习过程，推理时 actor 仍然只依赖本体感觉。**

- 真正起作用的不是更大的网络，而是把可解释物理量同时用在两个位置：特权 critic 的输入，以及奖励塑形信号。
- 这种非对称设计是它可落地的前提——若把这些量放进 actor 观测，部署就会绑死在难以在线获取的状态估计上。
- 结果侧的关键在"统一"而非单点性能：一个策略覆盖从小扰动到大跌倒后恢复的完整行为链，本页报告 93.4% 的恢复成功率。
- 适用边界：本页为索引级实体，93.4% 对应的实验条件、消融设置与真机验证细节须回到深读笔记与论文 PDF（见[参考来源](#参考来源)）。
- 本页归入 04_Loco-Manipulation_and_WBC，可经[分类父节点](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)与同类工作交叉检索。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_embedding-classical-balance-control-principles-i.md](../../sources/papers/humanoid_pnb_embedding-classical-balance-control-principles-i.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery.html>
- 论文：<https://arxiv.org/abs/2603.08619>

## 推荐继续阅读

- [机器人论文阅读笔记：Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery/Embedding_Classical_Balance_Control_Principles_in_RL_for_Humanoid_Recovery.html)
