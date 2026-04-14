---
type: concept
tags: [locomotion, gait, CPG, rhythm, legged-robots, mpc]
status: stable
---

# Gait Generation（步态生成）

**Gait Generation** 是腿式机器人运动控制中负责**决定步态模式（gait pattern）的模块**：确定各腿的支撑/摆动相时序、步频、步幅范围，为步位规划和质心轨迹优化提供时序框架。步态生成是运动速度、能耗和稳定性之间的权衡起点。

## 一句话定义

> 步态生成决定的是"腿的节律"——哪条腿先动、动多久、多快——是比步位规划更高层的时序编排，也是机器人从慢走到疾跑的关键调节器。

---

## 常见步态类型

| 步态 | 支撑腿数 | 速度范围 | 典型机器人 |
|------|---------|---------|-----------|
| **Static walk** | 3-4 | 慢速 | 工业臂式机器人、重型机械 |
| **Trot** | 2（对角） | 中速 | ANYmal, Spot, Unitree A1/Go1 |
| **Pace** | 2（同侧） | 中速 | 骆驼步态，稳定但侧倾大 |
| **Canter / Gallop** | 1-2（动态） | 高速 | MIT Cheetah, Stanford Doggo |
| **Bound** | 2（前后） | 高速 | 四足高速奔跑 |
| **Pronk** | 4 同时腾空 | 跳跃 | 极限速度 / 跨越障碍 |

**双足步态**额外包含：Walking（始终有支撑脚）、Running（有腾空相）、Jumping（单次腾空）。

---

## 主流方法

### 1. 中枢模式生成器（CPG, Central Pattern Generator）

- **生物来源**：脊椎动物脊髓中负责协调肢体节律运动的神经网络
- **数学实现**：耦合振荡器（Hopf oscillator、Matsuoka oscillator）
- **输出**：各腿的相位信号 → 调制摆动/支撑时序
- **优点**：天然节律性，平滑，对扰动有一定鲁棒性
- **局限**：参数调整复杂，与地形感知集成困难

```
腿 i 振荡器：  ẋᵢ = f(xᵢ, xⱼ, φᵢⱼ)
相位耦合约束：  φᵢⱼ = π (trot), 0 (pace), π/2 (walk)...
```

### 2. 参数化步态（时钟式控制）

- **核心思路**：手工设计步态调度器：给定速度命令 → 查表/插值得到步频、duty factor
- **代表实现**：legged_gym 里的 gait scheduler，Walk These Ways 的步态参数条件化
- **优点**：直接可调，易与 RL 策略结合（步态参数作为 condition vector）
- **局限**：步态切换边界需手工设计，不能自动发现最优步态

### 3. MPC 联合优化（接触序列优化）

- **核心思路**：把接触时序（哪只脚在哪段时间内接触地面）作为 MPC 的优化变量或调度参数
- **代表工作**：Bledt & Kim (MIT), Farbod Farshidian (ETH RSL)  
- **优点**：可根据地形和速度自适应调整步态
- **局限**：混合整数优化，计算量大，通常需要热启动和步态序列枚举

### 4. 数据驱动 / RL 步态发现

- **核心思路**：RL 策略在训练中自动涌现步态行为（不显式编码步态）
- **代表工作**：Rudin 2022（legged_gym），学出来的策略在中速时自然形成 trot
- **优点**：策略驱动，不需要手工设计步态切换
- **局限**：涌现步态可能在某些速度区间不稳定，难以保证特定步态约束

---

## 步态生成在控制栈中的位置

```
速度命令（高层导航）
      ↓
  步态生成（本页）  ←── 地形高度 / 速度自适应
      ↓
  步位规划（contact sequence）
      ↓
质心轨迹规划（DCM / MPC）
      ↓
  全身运动控制（WBC）
```

---

## 与 RL 步态的关系

RL 策略通常不显式区分步态生成和策略执行——步态由奖励函数间接诱导：
- **接触奖励**：对期望接触模式给正奖励（legged_gym 里的 `feet_air_time` 奖励）
- **节律惩罚**：对非规律步态给负奖励
- **命令条件化**（Walk These Ways）：速度命令 → 策略自动选择对应步态

---

## 参考来源

- [sources/papers/mpc.md](../../sources/papers/mpc.md) — ingest 档案（MPC 接触序列优化）
- Ijspeert, *Central Pattern Generators for Locomotion Control in Animals and Robots: A Review* (2008) — CPG 综述
- Bledt et al., *MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot* (2018)

---

## 关联页面

- [Locomotion](../tasks/locomotion.md) — 步态生成是 locomotion pipeline 的核心调度层
- [Footstep Planning](./footstep-planning.md) — 步位规划是步态生成的下游
- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 实现自适应步态优化
- [Capture Point / DCM](./capture-point-dcm.md) — 步态时序影响 DCM 稳定域
- [Legged Gym](../entities/legged-gym.md) — legged_gym 中的 gait scheduler 参数化实现

---

## 推荐继续阅读

- Ijspeert et al., *Dynamical movement primitives: learning attractor models for motor behaviors*
- Margolis et al., *Walk These Ways: Tuning Robot Walking with a Reward Curriculum*

## 一句话记忆

> 步态生成是腿式机器人的"节拍器"——它决定各腿的起落时序，让机器人在不同速度和地形下选择合适的运动节律，从慢步到奔跑都有对应的协调模式。
</content>
