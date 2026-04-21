---
type: method
tags: [rl, safety, control, cmdp, optimization]
status: complete
updated: 2026-04-20
related:
  - ./reinforcement-learning.md
  - ../concepts/control-barrier-function.md
  - ../concepts/safety-filter.md
  - ../formalizations/cmdp.md
  - ../queries/robot-policy-debug-playbook.md
sources:
  - ../../sources/papers/privileged_training.md
summary: "安全强化学习（Safe RL）旨在满足显式安全约束的前提下优化策略，常用方法包括 Lagrangian 优化、受限策略投影和安全层屏蔽。"
---

# Safe RL（安全强化学习）

**安全强化学习（Safe Reinforcement Learning）** 是强化学习的一个分支，其核心目标是在训练和执行过程中，确保智能体满足特定的安全约束（如不发生碰撞、不超过关节限位、不跌倒），而不仅仅是追求累积奖励的最大化。

## 为什么需要 Safe RL？

标准强化学习（Vanilla RL）通常通过 **Reward Shaping**（在奖励函数中加入惩罚项）来诱导安全行为。然而，这种做法存在显著弊端：
- **权重难调**：惩罚太轻会导致机器人冒险，惩罚太重会导致机器人为了躲避惩罚而选择“摆烂”（如原地不动）。
- **无法提供硬保证**：即使奖励很高，也无法严格证明机器人在所有情况下都不会违反安全边界。
- **训练期间的危险**：在真实机器人上训练时，早期的随机探索极易导致硬件损坏。

## 核心形式化：CMDP

Safe RL 最常用的数学框架是 **Constrained MDP (CMDP)**。与标准 MDP 相比，它引入了额外的成本函数 $C(s, a)$ 和阈值 $\hat{c}$，目标是在满足 $E[\sum c_t] \leq \hat{c}$ 的前提下最大化 $E[\sum r_t]$。

详见：[Constrained MDP](../formalizations/cmdp.md)

## 主要技术路线

### 1. 基于拉格朗日的方法 (Lagrangian-based)
将约束转化为惩罚项，并动态调整其权重（拉格朗日乘子 $\lambda$）：
$$ \max_{\theta} \min_{\lambda \geq 0} \mathcal{L}(\theta, \lambda) = J_R(\pi_\theta) - \lambda (J_C(\pi_\theta) - \hat{c}) $$
- **优点**：实现简单，可直接适配 PPO/SAC。
- **缺点**：收敛较慢，容易出现奖励与约束的剧烈震荡。

### 2. 受限策略优化 (Constrained Policy Optimization, CPO)
在策略更新步骤中直接施加约束，确保每一步更新后的策略都位于安全集合内。通常涉及求解一个受限的二次规划问题（Trust Region 策略）。

### 3. 安全层与屏蔽器 (Safety Layers & Shielding)
在 RL 策略输出后，添加一个基于解析模型（如 CBF）的过滤层：
- 如果原始动作安全，则直接执行。
- 如果原始动作危险，则将其投影到安全区域内的最近动作。
这种方法通常被称为 **Safety Filter**。

### 4. 恢复策略 (Recovery RL)
学习两个策略：一个负责完成任务（Task Policy），另一个负责在即将进入危险区域时接管机器人（Recovery Policy）。

## 在机器人中的典型应用

- **人形机器人平衡**：限制质心（CoM）必须留在支撑多边形内。
- **协作机器人操作**：在保证工作效率的同时，确保末端速度或力度不超过人体承受上限。
- **自动驾驶**：在变道或跟车时，确保与障碍物保持最小安全距离。

## 关联页面
- [Reinforcement Learning](./reinforcement-learning.md)
- [Control Barrier Function](../concepts/control-barrier-function.md)
- [Safety Filter](../concepts/safety-filter.md)
- [Constrained MDP](../formalizations/cmdp.md)
- [Query：机器人策略排障手册](../queries/robot-policy-debug-playbook.md)

## 参考来源
- Achiam, J., et al. (2017). *Constrained Policy Optimization*.
- Ray, A., et al. (2019). *Benchmarking Safe Deep Reinforcement Learning*.
- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md)
