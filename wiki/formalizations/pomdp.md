---
type: formalization
tags: [pomdp, rl, state-estimation, partial-observability, belief-state]
status: complete
related:
  - ./mdp.md
  - ../concepts/state-estimation.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/state_estimation.md
---

# POMDP（部分可观测马尔可夫决策过程）

**POMDP** 是 MDP 的扩展，适用于智能体**无法直接观测完整系统状态**的场景——几乎所有真实机器人系统都是 POMDP。

## 一句话定义

> POMDP = MDP + 观测模型。智能体看不到真实状态 $s$，只能获得含噪观测 $o$，需要维护信念状态 $b(s)$ 来作决策。

## 为什么重要

真实机器人传感器有噪声、延迟和遮挡：

- IMU 有漂移，无法直接得到精确姿态
- 相机看不到被遮挡的接触点
- 关节力矩传感器有噪声

用 MDP 假设"状态完全可知"是理想化的。POMDP 是更精确的数学框架，理解它有助于解释为什么需要**状态估计**、为什么需要**历史信息**、为什么**RNN/Transformer 策略**比无记忆策略更鲁棒。

## 形式化定义

POMDP 由七元组定义：

$$\mathcal{M} = (S, A, T, R, \Omega, O, \gamma)$$

| 元素 | 含义 |
|------|------|
| $S$ | 状态空间（真实物理状态，不可直接观测） |
| $A$ | 动作空间 |
| $T(s' \mid s, a)$ | 状态转移概率 |
| $R(s, a)$ | 奖励函数 |
| $\Omega$ | 观测空间 |
| $O(o \mid s', a)$ | 观测模型（在执行动作 $a$ 后处于状态 $s'$，产生观测 $o$ 的概率） |
| $\gamma$ | 折扣因子 |

## 信念状态

由于无法直接观测 $s$，智能体维护一个**信念状态**（belief state）：

$$b(s) = P(s_t = s \mid o_{1:t}, a_{1:t-1})$$

信念状态是对当前真实状态的概率分布。信念更新（Bayes 滤波器）：

$$b'(s') \propto O(o \mid s', a) \sum_s T(s' \mid s, a) b(s)$$

这正是 **EKF / 粒子滤波器** 所做的事情——在 POMDP 框架下的信念更新。

## POMDP → MDP 的近似策略

精确求解 POMDP 是 PSPACE-complete，实际中常用近似：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| **信念 MDP** | 把信念 $b$ 作为 MDP 的状态 | 低维状态空间 |
| **历史观测作为输入** | 给策略 $\pi(a \mid o_{1:t})$ 输入观测序列 | RNN/Transformer 策略 |
| **状态估计器 + MDP** | EKF/UKF 先估计 $\hat{s}$，再用 $\hat{s}$ 做 MDP | 人形机器人实践中最常用 |
| **特权训练** | 训练时用真实状态，推理时用估计状态 | Sim2Real：teacher-student 框架 |

## 在人形机器人中的应用

```
真实传感器读数 (o_t) → [EKF / 粒子滤波] → 估计状态 (ŝ_t) → [策略 π] → 动作 (a_t)
      ↑ 噪声、延迟                    ↑ POMDP 中的信念更新近似
```

机器人 RL 的核心挑战之一：策略在仿真器（近似 MDP）中训练，部署在真实环境（POMDP）中。Sim2Real gap 的一个主要来源就是这个"观测完全性"的差异。

### 特权训练（Privileged Training）

训练阶段：teacher policy 输入真实状态 $s$（完整 MDP）
部署阶段：student policy 输入估计状态 $\hat{s}$（近似信念）

这是目前人形机器人 RL 最主流的 POMDP 处理方式。

## 与 MDP 的关键区别

| 维度 | MDP | POMDP |
|------|-----|-------|
| 状态可观测 | ✅ 完全可观 | ❌ 部分可观 |
| 策略输入 | $\pi(a \mid s)$ | $\pi(a \mid b)$ 或 $\pi(a \mid o_{1:t})$ |
| 求解复杂度 | 多项式时间（有限状态） | PSPACE-complete |
| 实际处理方式 | 直接 RL | 状态估计 + RL，或 RNN 策略 |

## 常见坑

- **忽略 POMDP 结构**：把机器人问题建模为 MDP，策略输入含噪传感器数据，但不加历史，导致策略在噪声下抖动
- **信念维度爆炸**：信念状态维度 = 状态空间大小，连续高维状态下直接表示信念不可行
- **观测滞后**：真实系统中传感器有 10-20ms 延迟，POMDP 框架可以自然建模，但 MDP 框架忽略了这一点

## 参考来源

- Kaelbling, Littman & Cassandra, *Planning and Acting in Partially Observable Stochastic Domains* (1998) — POMDP 经典综述
- Thrun, Burgard & Fox, *Probabilistic Robotics* (2005) — 机器人 POMDP 与信念状态实践
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — 特权训练处理 POMDP 的经典案例

## 关联页面

- [MDP](./mdp.md) — POMDP 是 MDP 加观测模型的扩展
- [State Estimation](../concepts/state-estimation.md) — 状态估计是 POMDP 中信念更新的实际实现
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 在 POMDP 下需要历史观测或状态估计器
- [Sim2Real](../concepts/sim2real.md) — Sim2Real gap 的核心来源之一是 MDP（仿真）vs POMDP（真实）的差距
