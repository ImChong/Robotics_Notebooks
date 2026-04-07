# Model Predictive Control (MPC)

**模型预测控制**：一种基于滚动时域优化的控制方法，在每个时刻求解一个有限时域的最优控制问题，只执行第一步，然后重复。

## 一句话定义

不是一次性算出全局最优控制，而是**每一步都在线求解一个有限时域优化问题，只执行当前动作，然后重复**。

## 为什么重要

MPC 是机器人控制中最接近“万能控制框架”的东西——只要你能建模，就能控制。

核心优势：

- **处理约束**：关节限位、接触力限制、碰撞避免——都可以自然地塞进优化约束里
- **处理多目标**：同时处理平衡、跟踪、能耗等多个目标
- **处理非线性**：非线性 MPC 可以处理复杂动力学
- **处理时变系统**：每步重新计算，自然适应变化

代价是**计算量大**，实时性要求高。

## 核心工作原理

### 预测模型
首先需要一个动力学模型：

$$\dot{x} = f(x, u)$$

或者离散形式：

$$x_{k+1} = f(x_k, u_k)$$

这个模型不需要很精确（这也是 MPC 的一个优点），但越精确效果越好。

### 有限时域优化
在每个时刻 $t$：

1. **预测**：用模型预测未来 $N$ 步的状态序列
2. **优化**：求解一个优化问题，找到使代价函数最小的控制序列 $u_t, u_{t+1}, ..., u_{t+N-1}$
3. **执行**：只把 $u_t$ 发给机器人
4. **重复**：到下一个时刻，重新预测 + 求解

这就是"滚动时域"（receding horizon）控制的核心。

### 代价函数
典型形式：

$$J = \sum_{k=0}^{N-1} (x_k - x_{ref})^T Q (x_k - x_{ref}) + u_k^T R u_k$$

包含两项：
- **状态代价**：状态跟参考值的偏差
- **控制代价**：控制输入的大小（防止过于激进）

可以加硬约束（不等式）和软约束。

### 约束处理
这是 MPC 相比 LQR 等方法最关键的优势：

- **关节角度限位**：$\underline{q} \leq q \leq \bar{q}$
- **关节速度限位**：$\underline{\dot{q}} \leq \dot{q} \leq \bar{\dot{q}}$
- **接触力约束**：$f_z \geq 0$（地面反力非负）
- **碰撞避免**：作为不等式约束加入

## 主要类型

### 1. 线性 MPC（LMPC）
假设线性系统：

$$x_{k+1} = A x_k + B u_k$$

可以用 QP（二次规划）求解，速度快，适合实时控制。

典型应用：双足行走的 ZMP 控制、轮式机器人轨迹跟踪。

### 2. 非线性 MPC（NMPC）
用非线性动力学模型，需要求解非线性优化问题（NLP）。

计算量比 LMPC 大很多，通常需要：

- 实时非线性优化求解器（e.g. Acados, FORCES Pro）
- 比较好的初值（可以用线性 MPC 的结果做 warm start）

### 3. 凸非线性 MPC
把非线性问题凸化——比如用多面体约束、凸成本函数。

IsaacGym /机器人领域常见。

### 4. 随机 MPC
处理模型不确定性和外界干扰。

加入随机约束或者 robustness budget。

## 在人形机器人中的典型应用

### 行走控制
用 centroidal dynamics + NMPC：

- 预测未来几步的质心/角动量
- 优化接触力分配
- 保持平衡约束

代表工作：
- "Convex Model Predictive Control for Bipedal Locomotion" (C原 Bellicoso et al.)
- ANYmal 的行走控制

### 全身控制
用 WBC 框架下：

- 上层用 MPC 计算 task-space 指令
- 下层用 QP 分配关节力矩

### 足式机器人
四足/双足机器人的站立、行走、跑跳控制几乎都离不开 MPC。

代表：
- MIT Cheetah 的凸 MPC
- Unitree 的 NMPC 控制器

## 关键参数

### 预测时域 $N$
- $N$ 太短：来不及规划，响应快但可能不稳定
- $N$ 太长：计算量大，实时性差
- 通常 $N=10\sim 40$ 步（取决于控制频率）

### 控制频率
- 人形机器人典型：1-5 kHz
- 越快越好，但受限于求解器速度
- 高频MPC + 低频 RL 是常见组合

### 模型精度 vs 计算速度
这是一个 trade-off：
- 精确模型 → 更好的跟踪和预测，但计算慢
- 简化模型 → 计算快，但跟踪精度差

常用 trick：用简化模型做 MPC，用更精确的模型做仿真验证。

## 和强化学习的比较

| 维度 | MPC | 强化学习 |
|------|-----|---------|
| 依赖模型 | 必须有模型 | 不需要（model-free）|
| 计算量 | 在线优化，实时要求高 | 离线训练，在线推理快 |
| 泛化能力 | 受模型精度限制 | 可泛化到新任务 |
| 处理约束 | 自然 | 需要专门设计 |
| 超越人类 | 难 | 可以 |

常见组合：
- **RL 训练低层策略 + MPC 做高层规划**
- **RL 训练一个 "policy prior" + MPC 在其中做优化**

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Optimal Control (OCP)](../concepts/optimal-control.md)

## 推荐继续阅读

- [Optimal Control 2025 (YouTube Course)](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI) - Lecture 10: Convex Model-Predictive Control
- [Convex MPC for Bipedal Locomotion](https://arxiv.org/abs/1709.10219) - Cory (Bellicoso et al.)
- Acados: http://acados.org/ - NMPC solver
