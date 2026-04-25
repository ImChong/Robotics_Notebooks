---
type: formalization
tags: [hjb, optimal-control, continuous-time, dynamic-programming, value-function]
status: complete
related:
  - ./bellman-equation.md
  - ./lqr.md
  - ../methods/trajectory-optimization.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/optimal_control_theory.md
summary: "HJB 方程（Hamilton-Jacobi-Bellman）"
updated: 2026-04-25
---

# HJB 方程（Hamilton-Jacobi-Bellman）

**HJB 方程**是连续时间最优控制的基本方程，给出了最优值函数 $V^*(x,t)$ 满足的偏微分方程（PDE）。它是 Bellman 最优方程在连续时间域的推广。

## 一句话定义

> HJB = 连续时间版 Bellman 最优方程。满足 HJB 的值函数对应最优控制策略；求解 HJB 等价于解连续时间最优控制问题。

## 为什么重要

- **轨迹优化的理论基础**：MPC、Crocoddyl 等工具求解的是 HJB 的离散近似
- **控制证书**：Lyapunov 稳定性分析和 Control Barrier Function（CBF）都与 HJB 密切相关
- **LQR 是特例**：线性系统 + 二次代价下，HJB 方程有解析解，就是 LQR 的 Riccati 方程
- **理解 RL vs OC**：HJB（连续时间）和 Bellman（离散时间）对应同一思想，建立了 RL 与最优控制的理论桥梁

## 连续时间最优控制问题

有限时域问题：

$$V^*(x, t) = \min_{u(\cdot)} \left[ \int_t^T \ell(x(\tau), u(\tau))\, d\tau + \Phi(x(T)) \right]$$

约束：$\dot{x} = f(x, u)$（系统动力学），$V^*(x, T) = \Phi(x)$（终端代价）

## HJB 方程推导

对最优值函数 $V^*(x,t)$，通过 Bellman 最优性原理（最优子结构），在 $[t, t+dt]$ 上展开：

$$V^*(x,t) = \min_u \left[ \ell(x,u)\,dt + V^*(x + f(x,u)\,dt,\; t+dt) \right]$$

对右侧 Taylor 展开 $V^*$ 并取 $dt \to 0$，得到 **HJB 偏微分方程**：

$$-\frac{\partial V^*}{\partial t}(x,t) = \min_u \left[ \ell(x,u) + \nabla_x V^*(x,t)^\top f(x,u) \right]$$

边界条件：$V^*(x,T) = \Phi(x)$

### 无限时域（稳定控制）

$$0 = \min_u \left[ \ell(x,u) + \nabla_x V^*(x)^\top f(x,u) \right]$$

这等价于 Bellman 最优方程在 $\gamma \to 1$、连续时间极限的形式。

## 最优控制策略提取

一旦 $V^*$ 已知，最优控制律为：

$$u^*(x,t) = \arg\min_u \left[ \ell(x,u) + \nabla_x V^*(x,t)^\top f(x,u) \right]$$

这是一个**逐点最小化**——给定状态 $x$ 和 $\nabla V^*$（代价梯度），找最优 $u$。

## HJB 的特殊情况

### LQR 是 HJB 的解析特例

线性系统 $\dot{x} = Ax + Bu$，二次代价 $\ell(x,u) = x^\top Q x + u^\top R u$：

HJB 的解是二次型：$V^*(x) = x^\top P x$

其中 $P$ 满足**代数 Riccati 方程（ARE）**：

$$PA + A^\top P - PBR^{-1}B^\top P + Q = 0$$

最优控制：$u^*(x) = -R^{-1}B^\top P x$（线性状态反馈）。

### Hamilton 函数

定义 Hamilton 函数（Hamiltonian）：

$$H(x, u, p) = \ell(x,u) + p^\top f(x,u)$$

其中 $p = \nabla_x V^*$ 是**协态变量**（costate），HJB 变为：

$$-\frac{\partial V^*}{\partial t} = \min_u H(x, u, \nabla_x V^*)$$

Pontryagin 极大值原理（PMP）就是这个最小化条件的必要条件形式。

## HJB vs 离散 Bellman vs RL

| 框架 | 时域 | 求解方法 | 工具 |
|------|------|---------|------|
| HJB | 连续时间 | PDE 求解（难！高维不可行） | 解析（LQR）、数值（SOS、GP） |
| Bellman 最优 | 离散时间 | DP、值迭代 | 表格 RL、DQN |
| Deep RL | 离散时间 | 函数近似 + 采样 | PPO、SAC |
| MPC / DDP | 连续时间离散化 | 数值优化（SQP、iLQR） | Crocoddyl、acados |

**关键洞察**：MPC 在每个控制周期里求解一个短时域的 HJB 近似问题（离散化 + 近似终端代价）。

## 在机器人控制中的意义

```
连续时间动力学
  ↓ (离散化)
离散时间 MDP
  ↓ (函数近似)
Deep RL (PPO/SAC)
```

每一步都在用不同精度近似同一个 HJB 问题。理解 HJB 有助于：
1. 判断 RL 的理论解（HJB 最优）vs 实际近似解的差距
2. 设计终端代价函数（告诉 MPC "horizon 结束后还要花多少代价"）
3. 理解为什么 CBF（控制障碍函数）能提供安全保证——CBF 是 HJB 的 barrier 形式

## 求解困难与实际处理

**维度灾难**：HJB 是 PDE，对 $n$ 维状态空间需要在 $n$ 维空间上求解，计算量随维度指数增长（对人形机器人 30+ DOF 完全不可行）。

**实际解法**：
- **LQR**：线性系统解析解（小维度可用）
- **iLQR / DDP**：局部二阶近似，沿轨迹求解（MPC 框架）
- **MPC + 短时域**：只在短时域 $[t, t+T]$ 近似求解，用终端代价代替无限时域
- **Deep RL**：用神经网络近似 $V^*$，避免显式求解 PDE

## 参考来源

- Bellman, *Dynamic Programming* (1957) — 离散时间 Bellman 原理
- Fleming & Rishel, *Deterministic and Stochastic Optimal Control* (1975) — HJB 经典教材
- Kirk, *Optimal Control Theory: An Introduction* — 工程向 HJB 教材
- Tedrake, *Underactuated Robotics* (MIT OCW) — 机器人最优控制实践

## 关联页面

- [Bellman Equation](./bellman-equation.md) — HJB 是 Bellman 最优方程的连续时间极限
- [LQR](./lqr.md) — LQR 是 HJB 在线性二次系统上的解析解
- [Trajectory Optimization](../methods/trajectory-optimization.md) — 轨迹优化（iLQR/DDP）是 HJB 的数值近似方法
- [Optimal Control](../concepts/optimal-control.md) — HJB 和 Pontryagin PMP 是连续时间最优控制的两大理论工具
