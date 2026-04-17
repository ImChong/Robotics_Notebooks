---
type: formalization
tags: [control, lqr, optimal-control, linear-systems, locomotion]
status: complete
related:
  - ../concepts/optimal-control.md
  - ../methods/model-predictive-control.md
  - ../methods/trajectory-optimization.md
sources:
  - ../../sources/papers/optimal_control_theory.md
  - ../../sources/papers/optimal_control.md
---

# LQR / iLQR

**LQR（Linear Quadratic Regulator，线性二次调节器）**：最优控制中最经典的解析解，针对线性系统 + 二次代价函数，给出最优状态反馈增益的闭式解。**iLQR（iterative LQR）**是其非线性扩展，通过迭代线性化求解非线性轨迹优化。

## 一句话定义

> LQR 回答的是："对于一个线性系统，如果目标是最小化状态偏差和控制代价，最优的反馈控制律是什么？" iLQR 把这个框架推广到非线性系统，是轨迹优化的核心算法之一。

## 为什么重要

- LQR 是最优控制（OCP）的最简洁、最完备的例子：有解析解，有稳定性保证
- 理解 LQR 是理解 MPC、Riccati 方程、Bellman 最优方程的基础
- iLQR 是 DDP（Differential Dynamic Programming）的简化版，是 trajectory optimization 的主流算法之一，Crocoddyl 用的就是 iLQR/FDDP
- 在人形控制中：LQR 用于平衡稳定，iLQR 用于轨迹优化

## LQR 问题

### 系统模型

线性时不变系统：

$$x_{t+1} = A x_t + B u_t$$

或连续时间版本：

$$\dot{x} = Ax + Bu$$

### 代价函数

$$J = \sum_{t=0}^{T} \left( x_t^T Q x_t + u_t^T R u_t \right) + x_T^T Q_f x_T$$

其中：
- $Q \succeq 0$：状态代价矩阵（惩罚状态偏差）
- $R \succ 0$：控制代价矩阵（惩罚控制量大小）
- $Q_f$：终端状态代价

### 最优解：Riccati 方程

最优控制律是线性状态反馈：

$$u_t^* = -K_t x_t$$

其中增益矩阵 $K_t$ 由 **Riccati 方程** 反向递推求解：

$$P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

$$K_t = (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

对于无限时域 LQR（ILQR），Riccati 方程有稳态解 $P_\infty$，对应时不变增益 $K_\infty$。

### 稳定性保证

若 $(A, B)$ 可控，$(A, \sqrt{Q})$ 可观，则无限时域 LQR 解全局稳定（Lyapunov 意义下）。

## iLQR 问题

### 扩展到非线性系统

$$x_{t+1} = f(x_t, u_t)$$

$$J = \sum_{t=0}^{T} l(x_t, u_t) + l_f(x_T)$$

### 算法思路（Differential Dynamic Programming 简化版）

iLQR 是迭代算法，每轮包含：

**Backward pass（逆向传播）**：

1. 在当前轨迹 $\{x_t^*, u_t^*\}$ 处将系统线性化：$A_t = \partial f / \partial x$，$B_t = \partial f / \partial u$
2. 将代价函数二阶展开
3. 用 LQR 递推求解局部最优更新方向 $\delta u_t = K_t \delta x_t + k_t$

**Forward pass（正向传播）**：

用更新后的控制律 $u_t \leftarrow u_t^* + \alpha (K_t \delta x_t + k_t)$ 滚出新轨迹，更新参考轨迹

**迭代**直到收敛（代价不再下降）。

### 优势与局限

| 属性 | LQR | iLQR |
|------|-----|-------|
| 系统假设 | 线性 | 非线性（可微） |
| 求解复杂度 | $O(n^3)$ 每步 | $O(n^3 T)$ 每次迭代 |
| 全局最优 | ✅（线性系统） | ❌（局部最优） |
| 处理约束 | 需额外处理 | 需额外处理 |
| 实时性 | 离线增益 → 在线线性反馈 | 需要迭代，不直接在线 |

## 在机器人中的应用

### LQR 的典型用途
- 倒立摆稳定（入门 demo）
- 人形机器人躯干姿态稳定
- 线性化工作点附近的局部稳定控制
- 作为 MPC 的终端代价（terminal cost in MPC）

### iLQR 的典型用途
- 轨迹优化（连续接触动力学轨迹规划）
- Crocoddyl 中的 SolverFDDP / SolverBoxFDDP
- 全身运动生成（WBC 轨迹优化层）
- 人形行走摆腿轨迹设计

## LQR vs MPC 的关系

| 维度 | LQR | MPC |
|------|-----|-----|
| 预测窗口 | 无限时域（解析） | 有限时域（在线求解） |
| 处理约束 | 不直接支持 | 核心优势 |
| 非线性 | 需要线性化 | 可以处理非线性（NMPC） |
| 计算方式 | 离线 Riccati | 在线优化 |

MPC 在某种意义上是"在线、有约束、有限时域的 LQR"。

## 关联页面

- [Optimal Control (OCP)](../concepts/optimal-control.md) — LQR 是 OCP 的最经典闭式解
- [Trajectory Optimization](../methods/trajectory-optimization.md) — iLQR 是 TO 的核心算法
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — MPC 是 LQR 的有限时域、在线、带约束版本
- [Bellman 方程](./bellman-equation.md) — Riccati 方程是 Bellman 最优方程在线性二次问题上的解析形式
- [Crocoddyl](../entities/crocoddyl.md) — iLQR/FDDP 的开源实现

## 参考来源

- Anderson & Moore, *Optimal Control: Linear Quadratic Methods* — LQR 经典教材
- Todorov & Li, *A generalized iterative LQG method for locally-optimal feedback control* (2005) — iLQR 核心论文
- Tassa et al., *Synthesis and stabilization of complex behaviors through online trajectory optimization* (2012) — MuJoCo + iLQR 系统

## 推荐继续阅读

- [Optimal Control (OCP)](../concepts/optimal-control.md)
- [Trajectory Optimization](../methods/trajectory-optimization.md)
- [Crocoddyl](../entities/crocoddyl.md)（iLQR/FDDP 的工程实现）

## 一句话记忆

> LQR 是"线性系统最优反馈控制"的解析答案，Riccati 方程是它的核心工具；iLQR 是把这套框架推广到非线性系统的迭代算法，是轨迹优化的主力工具。
