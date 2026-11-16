---
type: method
tags: [rl, multi-agent, swarm, coordination]
status: complete
updated: 2026-11-16
related:
  - ./reinforcement-learning.md
  - ./rlcr-collective-inference-time-scaling.md
  - ../concepts/rl-runner.md
  - ../tasks/humanoid-soccer.md
  - ../concepts/humanoid-multi-robot-coordination.md
  - ../concepts/whole-body-coordination.md
  - ../concepts/whole-body-tracking-pipeline.md
  - ../entities/paper-assistmimic.md
  - ../entities/paper-humanoid-soccer-swarm-intelligence.md
  - ../entities/paper-rhythm-dual-humanoid-interaction.md
  - ../entities/paper-gamma-world-multi-agent.md
  - ../methods/generative-world-models.md
original:
  - ../../sources/blogs/gr_how_to_train_a_swarm_2026_11_02.md
sources:
  - ../../sources/blogs/gr_how_to_train_a_swarm_2026_11_02.md
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/papers/survey_papers.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
  - ../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md
summary: "多智能体强化学习（MARL）研究多个自主智能体在共享环境中的交互与进化，涵盖了竞争、协作及纳什均衡等复杂博弈动力学。"
---

# 多智能体强化学习（MARL） | Multi-Agent Reinforcement Learning

## 一句话定义

**MARL** 扩展了单智能体 RL，处理多个机器人在同一空间协作或竞争的问题（如机器人足球、多臂流水线）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| CTDE | Centralized Training, Decentralized Execution | 训练见全局、执行用局部观测的 MARL 范式 |
| ACO | Ant Colony Optimization | 非 RL 的群智能角色分配；与人形足球 swarm 论文相关 |

## 为什么重要

多机器人协作时，每台机器人的动作都可能改变其他机器人的观测、回报和可行动作。学习策略不仅要解决自己的动作选择，还要处理**其他策略同时变化**、团队目标与个人奖励可能不一致的问题。对机器人足球或双人协作，分工、信息共享和断联后的自主恢复都影响最终任务表现；对软件 Agent 团队，还需要考虑实际用时与交流开销（见 [RLCR](./rlcr-collective-inference-time-scaling.md)）。

## 核心原理

把多智能体环境抽象为：每个智能体 (i) 根据局部观测 (o_i) 输出动作 (a_i)；环境依据联合动作 ((a_1,dots,a_n)) 产生下一状态与回报。训练时需分别说明**信息边界**（集中式/去中心化）、**参数是否共享**和**回报归属**（个体/团队/混合），它们是不同设计维度。

### 两类常见信息组织形式



1. **集中式训练，分布式执行 (CTDE)**：训练时模型可以看到所有机器人的全局信息，部署时每个机器人仅根据局部观测决策（如 MAPPO）。
2. **完全去中心化**：每个机器人将其他机器人视为环境的一部分。

## 挑战
- **非平稳性 (Non-stationarity)**：由于其他机器人的策略在变，环境的转移概率也在变。
- **维度爆炸**：联合动作空间随机器人数量指数级增长。

## 工程实践

| 环节 | 机器人 / 软件群体应核验的条件 |
|---|---|
| 观测 | 部署时每个参与者实际能见到什么；训练时是否额外使用全局状态 |
| 奖励 | 队伍总分、个体回报和通信代价如何分别计量；共享奖励不能直接解释单体贡献 |
| 动作与同步 | 物理机器人检查执行周期、网络时延、丢包恢复；软件 Agent 检查超时、资源竞争与任务交接 |
| 基线 | 与独立智能体、集中式调度、启发式角色分配等对照；在同一环境和预算下比较 |
| 结果 | 除总成功率，还要记录任务时间、通信量、角色重分配延迟、规模变化后的效果 |

对照 [Sensors 2025 人形足球群智能论文摘录](../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md)，其 Webots 4v4 中角色重分配时间为 **2.3±0.4 s**（集中式对照为 **5.1±0.6 s**）：这是 **ACO + flocking** 系统的仿真数据，并非学习式 MARL 或 RLCR 的效果。

## 与生成式世界模型的交叉

传统 MARL 在 **解析或网格仿真** 里学联合策略；[Gamma-World](../entities/paper-gamma-world-multi-agent.md) 代表另一条路：先学 **多体动作条件的像素 rollout**（共享世界 + 各体独立可控），再作为 **MARL / 规划的想象环境**。二者互补——前者给 **策略梯度**，后者给 **高保真可交互观测**；是否进入闭环任务增益仍需单独验证（见 [训练闭环 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)）。

人形 **物理交互** 方向的近期实例：[AssistMimic](../entities/paper-assistmimic.md) 用 **联合 PPO** 学仿真双人 assistive tracking；[Rhythm](../entities/paper-rhythm-dual-humanoid-interaction.md) 用 **MAPPO + 图结构奖励** 在 **双 G1 真机** 实现拥抱/共舞等耦合行为——二者共享 **CTDE / partner-aware 观测** 叙事，但问题设定（护理 vs 对称社交）与 sim2real 栈不同。

## 与人形足球群控的关系

MARL 是人形 **多机战术** 的学习式路线之一（自博弈、CTDE 等），但 RoboCup 实战常受 **通信配额与样本效率** 约束。近期一手资料形成对照：

- **学习式：** 综述与 MARLadona 等指向 **局部观测 + 课程 + 自博弈**（见 [Swarm 人形足球](../entities/paper-humanoid-soccer-swarm-intelligence.md) §1.2 引文）。
- **规则式 swarm：** [Sensors 2025 人形足球 swarm](../entities/paper-humanoid-soccer-swarm-intelligence.md) 用 **ACO + flocking + 轻量 RL**，强调 **秒级在线重分配**，相对离线 MARL 更省数据。
- **工程拍卖：** [SPL 有限通信 arXiv:2401.15026](../../sources/papers/robocup_spl_limited_communication_coordination_arxiv_2401_15026.md) 用 **市场机制 + Voronoi**，在 **NAO 真机联赛** 验证。

选型总览见 [人形多机协调](../concepts/humanoid-multi-robot-coordination.md)。

## 与软件 Agent 群体 RLCR 的关系

[RLCR：集体奖励与集体推理时扩展](./rlcr-collective-inference-time-scaling.md)（General Reasoning，2026-11-02）讨论共享策略、多 Agent 统一回报、wall-clock 截止时间与通信原语如何促成**软件 Agent 的并行协作**。它与本页 MARL 共享多智能体奖励和信用分配问题，但其 Avalon-M1 结果仅来自长时软件任务；**不是机器人编队、多机动作控制或实体群体的真机验证**。迁移至机器人需要另外处理通信失效、安全约束与物理动力学。

## 局限与风险

- **非平稳与信用分配：** 所有参与者同时学习时，单体很难判断共同结果究竟来自谁；纯团队奖励可能掩盖偷懒、重复劳动或通信噪声。
- **训练与部署可观测性不一致：** CTDE 中训练者可用的全局信息未必能在真实机器人本地获得，部署时要明确传感和通信边界。
- **仿真不能直接代替真机结论：** 足球群智能示例是 Webots 仿真；实际网络丢包、运动学误差和安全限制需要单独验证。
- **软件群与实体群不可等同：** [RLCR](./rlcr-collective-inference-time-scaling.md) 的 wall-clock 截止时间及消息频道描述不等于已验证的机器人实时控制协议；其原文通读仅支撑本页该交叉部分，其他工作沿用已有资料归档。

## 关联页面
- [RLCR：集体奖励与集体推理时扩展](./rlcr-collective-inference-time-scaling.md) — 软件 Agent 群共享回报与延迟激励，不能直接等同机器人 MARL
- [CTDE vs 完全去中心化 MARL](../comparisons/ctde-vs-decentralized-marl.md) — 两大训练范式选型对比
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — Multi-agent / Self-play Runner：打包多 agent 观测或从策略池抽对手
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
- [General Reasoning: How to Train a Swarm（2026-11-02，已核对全文）](../../sources/blogs/gr_how_to_train_a_swarm_2026_11_02.md) — 支撑软件 Agent 共享集体回报、time limit 与通信机制对照；不用于证明机器人效果
- [humanoid_soccer_swarm_intelligence_sensors_2025.md](../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md)
- Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning*.

## 推荐继续阅读

- [How to Train a Swarm（General Reasoning 官方原文）](https://www.gr.inc/blog/how-to-train-a-swarm) — RLCR 集体回报、时间压力与通信的原始机制。
- [Swarm Intelligence for Collaborative Play in Humanoid Soccer Teams（Sensors 2025）](https://doi.org/10.3390/s25113496) — 机器人足球的规则式/轻量 RL 群控对照；不要把它算作纯 MARL 训练结果。
