---
type: comparison
tags: [rl, multi-agent, marl, ctde, mappo, coordination]
status: complete
related:
  - ../methods/marl.md
  - ../methods/reinforcement-learning.md
  - ../concepts/whole-body-coordination.md
  - ../entities/paper-gamma-world-multi-agent.md
sources:
  - ../../sources/papers/survey_papers.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
summary: "CTDE（集中式训练，分布式执行）vs 完全去中心化：多智能体 RL 两大训练范式的选型对比。"
updated: 2026-05-31
---

# CTDE vs 完全去中心化 MARL

在 [多智能体强化学习（MARL）](../methods/marl.md) 里，多个机器人在同一空间协作或竞争（多臂流水线、机器人足球、集群导航）。两个核心难点是 **非平稳性**（别人的策略在变，环境转移也在变）与 **联合动作空间维度爆炸**。围绕「训练时能看到多少信息」这条轴，MARL 分成两大范式：**CTDE** 与 **完全去中心化**。

## 核心对比

| 维度 | CTDE（集中式训练，分布式执行） | 完全去中心化 |
|------|-------------------------------|-------------|
| **训练期信息** | 可访问全局状态 / 所有体观测与动作 | 各体只见自身局部观测 |
| **执行期信息** | 仅局部观测（与去中心化一致） | 仅局部观测 |
| **非平稳性缓解** | ✅ 集中式 critic 看全局，缓解信用分配 | ❌ 把他体当环境的一部分，非平稳严重 |
| **可扩展性** | 受集中式 critic 输入维度限制 | 体数增长更平滑（各体独立） |
| **部署耦合度** | 低（执行期不需通信） | 低（天然分布式） |
| **代表算法** | MAPPO、MADDPG、QMIX | 独立 PPO/Q-learning（IPPO/IQL） |

## CTDE（Centralized Training, Decentralized Execution）

训练时用一个能看到 **全局信息** 的集中式 critic 估计价值，缓解信用分配与非平稳性；部署时每个机器人只凭 **局部观测** 决策，因此执行期无需体间通信。MAPPO 是把 PPO 套进 CTDE 框架的常见基线。

**适合**：体数中等、训练在仿真里能拿到全局状态、希望部署时各体独立运行的协作任务。

**瓶颈**：集中式 critic 的输入随体数增长，超大集群下维度与训练成本上升。

## 完全去中心化

每个机器人把 **其他机器人视为环境的一部分**，各自跑独立的单体 RL（IPPO/IQL）。实现简单、扩展平滑，但因为他体策略在变，单体看到的环境是 **非平稳** 的，收敛性与稳定性更差。

**适合**：体数很大、无法获取全局状态、或只需弱协作的场景。

**瓶颈**：非平稳性导致训练不稳，复杂协作任务上常不如 CTDE。

## 与生成式世界模型的交叉

传统 MARL 在 **解析或网格仿真** 里学联合策略；[Gamma-World](../entities/paper-gamma-world-multi-agent.md) 走另一条路：先学 **多体动作条件的像素 rollout**（共享世界 + 各体独立可控），再作为 MARL / 规划的 **想象环境**。二者互补——前者给策略梯度，后者给高保真可交互观测；是否带来闭环任务增益仍需单独验证。

## 选型小结

| 场景 | 推荐范式 | 原因 |
|------|---------|------|
| 中等体数、仿真有全局状态 | CTDE（MAPPO） | 集中式 critic 缓解非平稳与信用分配 |
| 大规模集群、无全局状态 | 完全去中心化 | 各体独立，扩展平滑 |
| 需部署时各体独立运行 | 两者皆可（执行期都只用局部观测） | 关键差异在训练期信息 |
| 强协作、信用分配难 | CTDE | 去中心化非平稳性放大协作难度 |

## 参考来源

- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.
- [sources/papers/survey_papers.md](../../sources/papers/survey_papers.md) — MARL 综述与范式分类
- [sources/papers/gamma_world_arxiv_2605_28816.md](../../sources/papers/gamma_world_arxiv_2605_28816.md) — 多智能体生成式交互世界模型

## 关联页面

- [Multi-Agent Reinforcement Learning (MARL)](../methods/marl.md) — 范式、挑战与分类总览
- [Reinforcement Learning](../methods/reinforcement-learning.md) — 单体 RL 基础
- [Whole-body Coordination](../concepts/whole-body-coordination.md) — 单体内部多自由度协调，与多体协调相对照
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — 多智能体生成式交互世界模型（arXiv:2605.28816）
