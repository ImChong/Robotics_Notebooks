# Optimal Control (OCP)

**最优控制**：给定一个动力学系统和一个代价函数，求解在有限或无限时域内使得代价最小的控制输入序列的理论框架。

## 一句话定义

“给定一个会动的机器人，什么样的控制序列能让它完成任务的同时代价最小？”

## 为什么重要

最优控制是现代控制理论的基石，也是 MPC、RL、WBC 的理论根基。

它解决的问题本质是：

> **找到一个控制策略，使得"做这件事花多少代价"最小。**

这个代价可以是：
- 能量消耗
- 跟踪误差
- 时间
- 或者它们的加权组合

在人形机器人里，最优控制的思路无处不在：
- **MPC**：在线滚动求解 OCP，是目前最主流的人形控制器架构
- **WBC**：底层 QP 等价于一个带约束的凸优化 OCP
- **RL**：model-free 最优控制——不显式建模，但目标同样是解 Bellman 最优方程
- **轨迹优化**：离线求解 OCP，给 MPC 提供参考轨迹或初始化

不懂 OCP，就看不懂当前人形控制算法的底层逻辑。

## 核心问题：OCP 的标准形式

一个标准最优控制问题（OCP）包括：

### 1. 动力学约束
$$x_{k+1} = f(x_k, u_k), \quad x \in \mathbb{R}^n, u \in \mathbb{R}^m$$

状态转移方程，描述机器人怎么动。

### 2. 代价函数
$$J = \sum_{k=0}^{N-1} g(x_k, u_k) + g_N(x_N)$$

包括：
- **阶段代价** $g(x_k, u_k)$：每一步的代价，如跟踪误差 + 控制努力
- **终端代价** $g_N(x_N)$：到达最终状态的好坏

### 3. 约束
$$x_k \in \mathcal{X}, \quad u_k \in \mathcal{U}$$

状态和控制都有约束，比如关节限位、接触力限制。

### 目标
$$\min_{u_0, ..., u_{N-1}} J$$

找到使总代价最小的控制序列。

## 主要求解方法

### 1. 变分法 / Pontryagin 极大值原理
经典方法。

引入 **协态变量** $\lambda$，构造哈密顿量：

$$H(x, u, \lambda) = g(x, u) + \lambda^T f(x, u)$$

求解一阶必要条件得到最优控制。

结果：**最优控制是正交条件下的"最优"动作**。

优点是有解析解，缺点是只适合线性和简单非线性问题。

### 2. 动态规划（DP）
从终点往回推。

定义 **值函数**（Value Function）：

$$V_k(x_k) = \min_{u_k, ..., u_{N-1}} \sum_{i=k}^{N-1} g(x_i, u_i) + g_N(x_N)$$

最优控制：

$$u_k^* = \arg\min_u [g(x_k, u) + V_{k+1}(f(x_k, u))]$$

问题：**维度灾难**——状态空间一大，DP 就不可行。

### 3. 数值优化方法
把 OCP 变成大规模数值优化问题来解。

适合复杂系统、非线性、约束多的情况。

典型方法：
- **直接法（Direct Methods）**：把连续最优控制问题离散化，然后用 NLP 求解
- **间接法（Indirect Methods）**：先求解析条件，再数值求解
- **打靶法（Shooting Methods）**：只优化控制序列，用模型前向传播验证
- **多打靶（Multiple Shooting）**：把状态也作为优化变量，解决打靶法的数值不稳定性

### 4. 强化学习 vs 最优控制

| | 传统最优控制 | 强化学习 |
|--|------------|---------|
| 模型 | 需要（白盒）| 不需要（model-free）|
| 求解 | 数值优化/解析 | RL 算法（策略梯度、Q学习等）|
| 泛化 | 受模型限制 | 可泛化 |
| 样本效率 | 高（一次优化）| 低（需要大量交互）|
| 适用场景 | 模型精确、实时性要求高 | 模型难以获得、任务复杂 |

关系：RL 可以看成"model-free 的最优控制"，DP 是 RL 的理论基础。

## 在机器人中的典型应用

### 轨迹优化
给定起点终点，求最优轨迹。

典型做法：
- 直接方法：把轨迹离散化，求 NLP
- 常用工具：TrajOpt, CHOMP, STOMP

### MPC（模型预测控制）
OCP 的在线滚动求解版本，是**最常用**的机器人实时控制框架。

见 [Model Predictive Control (MPC)](../methods/model-predictive-control.md)

### LQR / iLQR
特殊情形：线性系统 + 二次代价。

- LQR：无限时域线性二次调节器，有解析解
- iLQR：非线性系统的迭代 LQR，用于轨迹优化

### 最优动作规划
机械臂、移动机器人的路径/动作规划。

## 和 WBC 的关系

WBC（全身控制）本质上是一个**分层最优控制**：

```
任务空间目标（末端执行器轨迹）
       ↓
全身 QP 优化（力矩分配）
       ↓
关节力矩指令
```

其中：
- 上层可以用 MPC
- 下层 QP 是凸优化，等价于一种特殊的最优控制

## 常见坑

### 维度灾难
状态空间太大时 DP 不可行，需要用近似方法（function approximation → RL）。

### 非凸代价
很多 OCP 是非凸的，容易陷入局部最优。

常用 tricks：多初始点、random restarts、convex relaxation。

### 约束处理
硬约束 vs 软约束的设计对求解器的鲁棒性影响很大。

### 实时性
OCP 在线求解的计算量是大问题，尤其非线性 MPC。

解法：预计算、凸近似、定制求解器（Acados, FORCES Pro）。

## 参考来源

- Kirk, *Optimal Control Theory: An Introduction* — OCP 经典入门教材
- Bertsekas, *Dynamic Programming and Optimal Control* — DP 与 OCP 理论基础
- Betts, *Practical Methods for Optimal Control Using Nonlinear Programming* — 直接法数值实现参考

## 关联页面

- [LQR / iLQR](../formalizations/lqr.md) — LQR 是 OCP 在线性系统上的解析解；iLQR 是非线性扩展
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)
- [Whole-Body Control](./whole-body-control.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [MDP](../formalizations/mdp.md) — OCP 是确定性版本的 MDP；OCP 不含随机转移，RL 含随机转移
- [Bellman 方程](../formalizations/bellman-equation.md) — Bellman 最优方程是 OCP（尤其是 LQR）的解析求解基础

## 推荐继续阅读

- [Optimal Control 2025 (YouTube)](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI)
- 《Optimal Control Theory: An Introduction》- Kirk
- 《Robotics: Modelling, Planning and Control》- Siciliano
