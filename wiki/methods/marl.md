---
type: method
tags: [rl, multi-agent, swarm, coordination]
status: complete
updated: 2026-06-12
related:
  - ./reinforcement-learning.md
  - ../tasks/humanoid-soccer.md
  - ../concepts/humanoid-multi-robot-coordination.md
  - ../concepts/whole-body-coordination.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../entities/paper-assistmimic.md
  - ../entities/paper-humanoid-soccer-swarm-intelligence.md
  - ../entities/paper-rhythm-dual-humanoid-interaction.md
  - ../entities/paper-gamma-world-multi-agent.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/papers/survey_papers.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
  - ../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md
summary: "多智能体强化学习（MARL）研究多个自主智能体在共享环境中的交互与进化，涵盖了竞争、协作及纳什均衡等复杂博弈动力学。"
---

# Multi-Agent Reinforcement Learning (MARL)

**MARL** 扩展了单智能体 RL，处理多个机器人在同一空间协作或竞争的问题（如机器人足球、多臂流水线）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CTDE | Centralized Training, Decentralized Execution | 训练见全局、执行用局部观测的 MARL 范式 |
| ACO | Ant Colony Optimization | 非 RL 的群智能角色分配；与人形足球 swarm 论文相关 |

## 主要分类

1. **集中式训练，分布式执行 (CTDE)**：训练时模型可以看到所有机器人的全局信息，部署时每个机器人仅根据局部观测决策（如 MAPPO）。
2. **完全去中心化**：每个机器人将其他机器人视为环境的一部分。

## 挑战
- **非平稳性 (Non-stationarity)**：由于其他机器人的策略在变，环境的转移概率也在变。
- **维度爆炸**：联合动作空间随机器人数量指数级增长。

## 与生成式世界模型的交叉

传统 MARL 在 **解析或网格仿真** 里学联合策略；[Gamma-World](../entities/paper-gamma-world-multi-agent.md) 代表另一条路：先学 **多体动作条件的像素 rollout**（共享世界 + 各体独立可控），再作为 **MARL / 规划的想象环境**。二者互补——前者给 **策略梯度**，后者给 **高保真可交互观测**；是否进入闭环任务增益仍需单独验证（见 [训练闭环 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)）。

人形 **物理交互** 方向的近期实例：[AssistMimic](../entities/paper-assistmimic.md) 用 **联合 PPO** 学仿真双人 assistive tracking；[Rhythm](../entities/paper-rhythm-dual-humanoid-interaction.md) 用 **MAPPO + 图结构奖励** 在 **双 G1 真机** 实现拥抱/共舞等耦合行为——二者共享 **CTDE / partner-aware 观测** 叙事，但问题设定（护理 vs 对称社交）与 sim2real 栈不同。

## 与人形足球群控的关系

MARL 是人形 **多机战术** 的学习式路线之一（自博弈、CTDE 等），但 RoboCup 实战常受 **通信配额与样本效率** 约束。近期一手资料形成对照：

- **学习式：** 综述与 MARLadona 等指向 **局部观测 + 课程 + 自博弈**（见 [Swarm 人形足球](../entities/paper-humanoid-soccer-swarm-intelligence.md) §1.2 引文）。
- **规则式 swarm：** [Sensors 2025 人形足球 swarm](../entities/paper-humanoid-soccer-swarm-intelligence.md) 用 **ACO + flocking + 轻量 RL**，强调 **亚秒在线重分配**，相对离线 MARL 更省数据。
- **工程拍卖：** [SPL 有限通信 arXiv:2401.15026](../../sources/papers/robocup_spl_limited_communication_coordination_arxiv_2401_15026.md) 用 **市场机制 + Voronoi**，在 **NAO 真机联赛** 验证。

选型总览见 [人形多机协调](../concepts/humanoid-multi-robot-coordination.md)。

## 关联页面
- [CTDE vs 完全去中心化 MARL](../comparisons/ctde-vs-decentralized-marl.md) — 两大训练范式选型对比
- [Reinforcement Learning](./reinforcement-learning.md)
- [Whole-body Coordination](../concepts/whole-body-coordination.md)
- [AssistMimic](../entities/paper-assistmimic.md) — 双人 assistive MARL tracking（arXiv:2603.11346）
- [Rhythm](../entities/paper-rhythm-dual-humanoid-interaction.md) — 双 G1 真机交互 MAPPO + 图奖励（arXiv:2603.02856）
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — 多智能体生成式交互世界模型（arXiv:2605.28816）
- [Generative World Models](./generative-world-models.md) — 像素级世界模型总览
- [Humanoid Soccer](../tasks/humanoid-soccer.md) — 多机战术任务语境
- [人形多机协调](../concepts/humanoid-multi-robot-coordination.md) — 群控范式对比
- [Swarm Intelligence 人形足球](../entities/paper-humanoid-soccer-swarm-intelligence.md) — ACO+flocking 非 MARL 群控

## 参考来源
- [humanoid_soccer_swarm_intelligence_sensors_2025.md](../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md)
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.
