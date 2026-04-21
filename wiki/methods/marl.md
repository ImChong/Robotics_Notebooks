---
type: method
tags: [rl, multi-agent, swarm, coordination]
status: complete
updated: 2026-04-21
related:
  - ./reinforcement-learning.md
  - ../concepts/whole-body-coordination.md
sources:
  - ../../sources/papers/survey_papers.md
summary: "多智能体强化学习（MARL）研究多个自主智能体在共享环境中的交互与进化，涵盖了竞争、协作及纳什均衡等复杂博弈动力学。"
---

# Multi-Agent Reinforcement Learning (MARL)

**MARL** 扩展了单智能体 RL，处理多个机器人在同一空间协作或竞争的问题（如机器人足球、多臂流水线）。

## 主要分类

1. **集中式训练，分布式执行 (CTDE)**：训练时模型可以看到所有机器人的全局信息，部署时每个机器人仅根据局部观测决策（如 MAPPO）。
2. **完全去中心化**：每个机器人将其他机器人视为环境的一部分。

## 挑战
- **非平稳性 (Non-stationarity)**：由于其他机器人的策略在变，环境的转移概率也在变。
- **维度爆炸**：联合动作空间随机器人数量指数级增长。

## 关联页面
- [Reinforcement Learning](./reinforcement-learning.md)
- [Whole-body Coordination](../concepts/whole-body-coordination.md)

## 参考来源
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.
