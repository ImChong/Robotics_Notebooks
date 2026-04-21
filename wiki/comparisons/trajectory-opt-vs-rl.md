---
type: comparison
tags: [control, optimization, rl, trajectory-optimization, decision-making]
status: complete
updated: 2026-04-20
related:
  - ../methods/trajectory-optimization.md
  - ../methods/reinforcement-learning.md
  - ../methods/model-predictive-control.md
  - ../queries/when-to-use-wbc-vs-rl.md
summary: "轨迹优化（Trajectory Optimization）与强化学习（RL）的对比：前者依赖精确动力学模型求解开环最优解，后者通过数据驱动学习具有鲁棒性的闭环策略。"
---

# Trajectory Optimization vs Reinforcement Learning

在足式机器人运动控制领域，**轨迹优化 (Trajectory Optimization, TO)** 和 **强化学习 (Reinforcement Learning, RL)** 是两种截然不同但又互补的技术路线。

## 核心对比

| 维度 | Trajectory Optimization (TO) | Reinforcement Learning (RL) |
|------|------------------------------|-----------------------------|
| **理论基础** | 最优控制理论 (Optimal Control) | 马尔可夫决策过程 (MDP) |
| **依赖项** | 精确的解析动力学模型 | 仿真环境或真实交互数据 |
| **计算时机** | 通常在线实时求解 (MPC) 或离线规划 | 离线训练，在线极速推理 |
| **反馈形式** | 依赖外层反馈 (MPC) 或解析梯度 | 天然的闭环神经网络策略 |
| **泛化性** | 依赖模型准确度，对未知扰动敏感 | 通过域随机化 (DR) 具有极强鲁棒性 |
| **物理一致性** | ✅ 严格满足动力学约束 | ⚠️ 依赖奖励函数诱导，可能违反约束 |

## 轨迹优化的优势与局限

- **优势**：
  - 物理意义明确，动作精确可控。
  - 不需要冗长的训练过程，只要模型对，立刻能用。
  - 能够处理极其复杂的硬约束（如避障、接触力限位）。
- **局限**：
  - 容易陷入局部最优，需要良好的初始猜测。
  - 对“模型误差”（如非线性摩擦、电机非线性）非常敏感。

## 强化学习的优势与局限

- **优势**：
  - 能够发现人类难以设计的奇异步态（如通过摔倒恢复平衡）。
  - 在大时延、高噪声环境下表现更鲁棒。
  - 推理速度极快，适合部署在低算力嵌入式端。
- **局限**：
  - “黑盒”特性，安全性难以解释和证明。
  - 奖励函数（Reward Design）的设计非常主观，容易出现薅羊毛行为。

## 融合趋势：Learning to Optimize

现代前沿研究（如 ANYmal, Atlas 的最新版本）正在将两者融合：
1. **RL 辅助 TO**：用 RL 学习一个良好的初始轨迹值或价值函数，作为 TO 的热启动。
2. **TO 引导 RL**：利用 TO 产生的最优轨迹作为模仿学习的专家数据，加速 RL 训练。
3. **WBC + RL**：上层用 RL 产生任务目标，下层用 WBC (基于 QP 的 TO) 保证物理安全性。

## 关联页面
- [Trajectory Optimization (TO)](../methods/trajectory-optimization.md)
- [Reinforcement Learning (RL)](../methods/reinforcement-learning.md)
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)
- [Query：何时用 WBC vs RL？](../queries/when-to-use-wbc-vs-rl.md)

## 参考来源
- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md)
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md)
- Siciliano, B., et al. (2009). *Robotics: Modelling, Planning and Control*.
