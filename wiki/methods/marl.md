---
type: method
tags: [rl, multi-agent, swarm, coordination]
status: complete
updated: 2026-06-03
related:
  - ./reinforcement-learning.md
  - ../concepts/whole-body-coordination.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../entities/paper-assistmimic.md
  - ../entities/paper-gamma-world-multi-agent.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/papers/survey_papers.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
summary: "多智能体强化学习（MARL）研究多个自主智能体在共享环境中的交互与进化，涵盖了竞争、协作及纳什均衡等复杂博弈动力学。"
---

# Multi-Agent Reinforcement Learning (MARL)

**MARL** 扩展了单智能体 RL，处理多个机器人在同一空间协作或竞争的问题（如机器人足球、多臂流水线）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 主要分类

1. **集中式训练，分布式执行 (CTDE)**：训练时模型可以看到所有机器人的全局信息，部署时每个机器人仅根据局部观测决策（如 MAPPO）。
2. **完全去中心化**：每个机器人将其他机器人视为环境的一部分。

## 挑战
- **非平稳性 (Non-stationarity)**：由于其他机器人的策略在变，环境的转移概率也在变。
- **维度爆炸**：联合动作空间随机器人数量指数级增长。

## 与生成式世界模型的交叉

传统 MARL 在 **解析或网格仿真** 里学联合策略；[Gamma-World](../entities/paper-gamma-world-multi-agent.md) 代表另一条路：先学 **多体动作条件的像素 rollout**（共享世界 + 各体独立可控），再作为 **MARL / 规划的想象环境**。二者互补——前者给 **策略梯度**，后者给 **高保真可交互观测**；是否进入闭环任务增益仍需单独验证（见 [训练闭环 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)）。

## 关联页面
- [CTDE vs 完全去中心化 MARL](../comparisons/ctde-vs-decentralized-marl.md) — 两大训练范式选型对比
- [Reinforcement Learning](./reinforcement-learning.md)
- [Whole-body Coordination](../concepts/whole-body-coordination.md)
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — 多智能体生成式交互世界模型（arXiv:2605.28816）
- [Generative World Models](./generative-world-models.md) — 像素级世界模型总览

## 参考来源
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.
