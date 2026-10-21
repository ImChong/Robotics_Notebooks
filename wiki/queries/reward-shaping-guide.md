---
type: query
tags: [rl, reward-shaping, locomotion, training, optimization]
status: complete
updated: 2026-10-18
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

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |

## 核心框架：奖励项拆解

一个成熟的足式机器人奖励函数通常由三部分组成：**任务目标（Task）**、**运动质量（Smoothness）** 和 **安全性/约束（Safety）**。

### 1. 任务目标项 (The "Goal") — 引导机器人动起来
这些项通常是正奖励，或者是对目标速度的跟踪项。
- **速度跟踪**：$r_{vel} = \exp(-\frac{(v - v_{cmd})^2}{\sigma})$。指数核将奖励限制在有限范围内；带宽 $\sigma$ 决定有效学习区间，不能据此保证优于均方误差（MSE）。
- **角速度跟踪**：$r_{yaw} = \exp(-\frac{(\omega_z - \omega_{cmd})^2}{\sigma})$。

### 2. 运动质量项 (The "Style") — 引导步态看起来像个机器人
如果不加这些惩罚，机器人可能会出现高频振荡、内八字或疯狂抖动关节的情况。
- **机械功率（Power）**：可惩罚 $\sum_i |\tau_i\dot{q}_i|$；有符号的 $\sum_i\tau_i\dot{q}_i$ 会出现正负抵消。机械功率也不等于包含电机铜耗的电功率。
- **动作变化（Action Rate）**：惩罚 $\|a_t - a_{t-1}\|^2$；它约束策略动作变化，不等于关节加速度。关节加速度需由速度差分或仿真状态计算。
- **接触冲击**：可惩罚过大的足端接触力或落地速度；它与 Air Time 奖励衡量不同物理量，应分别调节。

### 3. 安全性项 (The "Boundary") — 引导机器人不坏掉
- **跌倒惩罚**：检测到躯干高度过低或非足端部位接地时给一个巨大的负值。
- **关节限位惩罚**：当 $q$ 接近 $q_{min}/q_{max}$ 时施加非线性惩罚。
- **力矩限位惩罚**：防止电机过载。

---

## 避坑与进阶技巧

### 1. 不要滥用正奖励
过多的正奖励会导致机器人学会“薅羊毛”（例如原地疯狂踩步而不前进以赚取 Air Time 奖励）。应根据任务目标选取奖励符号，并检查终止条件、站立策略和奖励投机；正负奖励的数量本身不是质量指标。

### 2. Air Time 奖励的艺术
`Feet Air Time` 可用于塑造摆动时长，但单独加入它不能保证出现跑步。
- 给摆动脚在空中的时间给正奖励。
- 可在首次落地时结算，并结合命令速度门控与目标摆动时长；是否跳跃仍取决于总奖励和动力学。

### 3. 奖励缩放 (Scaling)
- 不要让单项奖励完全支配总和。
- 记录每项原始值及加权后对总回报的贡献，再调权重；不存在通用的 50% / 30% / 20% 配比。改变控制周期时还需核对奖励是否乘以时间步长。

### 4. Curriculum（课程学习）
如果任务太难（如跑酷），不要一上来就加所有奖励。
- 先只奖励“不跌倒”。
- 稳定后再引入“速度跟踪”。
- 最后再优化“姿态平滑”。

---

## 奖励配置起点（示例）

以下仅作初始量级参考，不是可直接迁移的默认参数；需核对具体实现、动作缩放、机器人尺寸和控制周期。

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
