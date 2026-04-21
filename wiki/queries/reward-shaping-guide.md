---
type: query
tags: [rl, reward-shaping, locomotion, training, optimization]
status: complete
updated: 2026-04-20
related:
  - ../methods/reinforcement-learning.md
  - ../concepts/reward-design.md
  - ../queries/humanoid-rl-cookbook.md
  - ../queries/locomotion-failure-modes.md
sources:
  - ../../sources/papers/policy_optimization.md
summary: "Locomotion RL 奖励函数设计指南：通过将任务目标拆分为正向推进项、平滑惩罚项与安全性约束项，实现鲁棒且自然步态的技巧。"
---

# Locomotion RL 奖励函数设计指南

> **Query 产物**：本页由以下问题触发：「训练 locomotion RL 策略时，奖励函数怎么设计？有哪些调教技巧？」
> 综合来源：[Reward Design](../concepts/reward-design.md)、[Humanoid RL Cookbook](./humanoid-rl-cookbook.md)、[Legged Gym](../entities/legged-gym.md)

---

## 核心框架：奖励项拆解

一个成熟看足式机器人奖励函数通常由三部分组成：**任务目标（Task）**、**运动质量（Smoothness）** 和 **安全性/约束（Safety）**。

### 1. 任务目标项 (The "Goal") — 引导机器人动起来
这些项通常是正奖励，或者是对目标速度的跟踪项。
- **速度跟踪**：$r_{vel} = \exp(-\frac{(v - v_{cmd})^2}{\sigma})$。使用指数核比均方误差（MSE）收敛更平滑。
- **角速度跟踪**：$r_{yaw} = \exp(-\frac{(\omega_z - \omega_{cmd})^2}{\sigma})$。

### 2. 运动质量项 (The "Style") — 引导步态看起来像个机器人
如果不加这些惩罚，机器人可能会出现高频振荡、内八字或疯狂抖动关节的情况。
- **关节功耗（Energy）**：惩罚 $\tau \cdot \dot{q}$，防止过度用力。
- **关节加速度（Action Rate）**：惩罚 $\|a_t - a_{t-1}\|^2$，确保动作连续性。
- **足端平滑度**：惩罚足端接地时的冲击力（Air Time 奖励的对立面）。

### 3. 安全性项 (The "Boundary") — 引导机器人不坏掉
- **跌倒惩罚**：检测到躯干高度过低或非足端部位接地时给一个巨大的负值。
- **关节限位惩罚**：当 $q$ 接近 $q_{min}/q_{max}$ 时施加非线性惩罚。
- **力矩限位惩罚**：防止电机过载。

---

## 避坑与进阶技巧

### 1. 不要滥用正奖励
过多的正奖励会导致机器人学会“薅羊毛”（例如原地疯狂踩步而不前进以赚取 Air Time 奖励）。**原则是：除了速度跟踪，尽量使用惩罚项（负奖励）来约束行为。**

### 2. Air Time 奖励的艺术
`Feet Air Time` 是让机器人从“走”变“跑”的关键。
- 给摆动脚在空中的时间给正奖励。
- 注意：必须配合“落地后才结算奖励”的逻辑，否则机器人会学会双脚离地跳跃。

### 3. 奖励缩放 (Scaling)
- 不要让单项奖励完全支配总和。
- 典型的权重比例：速度跟踪占 50%，平滑项占 30%，安全性约束占 20%。

### 4. Curriculum（课程学习）
如果任务太难（如跑酷），不要一上来就加所有奖励。
- 先只奖励“不跌倒”。
- 稳定后再引入“速度跟踪”。
- 最后再优化“姿态平滑”。

---

## 典型奖励配置表 (参考 Legged Gym)

| 奖励项 | 类型 | 典型权重 | 目的 |
|--------|------|----------|------|
| `lin_vel_xy` | Positive | 1.0 - 2.0 | 跟踪前进速度 |
| `ang_vel_z` | Positive | 0.5 - 1.0 | 跟踪转弯速度 |
| `orientation` | Negative | -0.2 | 保持躯干水平 |
| `torques` | Negative | -0.0001 | 节能、保护电机 |
| `action_rate` | Negative | -0.01 | 动作平滑，防抖 |
| `feet_air_time` | Positive | 1.0 | 诱导产生节奏性步态 |
| `collision` | Negative | -1.0 | 防碰撞 |

---

## 关联页面
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Reward Design 概念](../concepts/reward-design.md)
- [Humanoid RL Cookbook](./humanoid-rl-cookbook.md)
- [Locomotion RL 失败模式分析](./locomotion-failure-modes.md)

## 参考来源
- Rudin, N., et al. (2022). *Learning to Walk in Minutes*.
- Margolis, G., et al. (2023). *Walk These Ways*.
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md)
