---
type: formalization
tags: [optimization, sensitivity, trajectory-optimization, automatic-differentiation]
status: complete
updated: 2026-06-23
related:
  - ../methods/trajectory-optimization.md
  - ../methods/lqr-ilqr.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "伴随灵敏度分析：高效计算 OCP/TrajOpt 对参数或初值的梯度，是 NLP 求解与系统辨识的核心工具。"
---

# Adjoint Sensitivity Analysis（伴随灵敏度分析）

**伴随灵敏度分析**：通过反向积分伴随方程（adjoint equation），以 $O(T)$ 代价计算长时域 OCP 目标对参数/初值/控制序列的梯度，避免有限差分的 $O(nT)$ 开销。

## 一句话定义

> 前向 rollout 存轨迹，反向解伴随 ODE 累积梯度——TrajOpt、NMPC 参数调优和可微仿真都依赖这一套。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCP | Optimal Control Problem | 伴随方程来自 OCP 的 Lagrangian |
| NLP | Nonlinear Programming | 离散 OCP 即有限维 NLP |
| AD | Automatic Differentiation | 自动微分，与伴随等价或互补 |
| iLQR | iterative Linear Quadratic Regulator | 反向 pass 与伴随结构同构 |
| NMPC | Nonlinear Model Predictive Control | 实时场景需高效梯度 |

## 为什么重要

- **TrajOpt 收敛**：L-BFGS / SQP 需要准确 $\nabla J$。
- **参数辨识**：$\partial J / \partial \theta$ 用于系统 ID、代价权重学习。
- **与 iLQR 同构**：iLQR backward pass 即离散伴随的一种实现。

## 连续时间形式（简述）

OCP：$\min \int_0^T L(x,u) dt + \Phi(x(T))$ s.t. $\dot{x}=f(x,u)$。

其中 $J$ 为总代价泛函，$T$ 为有限时域终端时刻，$x$ 为状态、$u$ 为控制。

定义 Hamiltonian $H = L + \lambda^T f$，伴随变量 $\lambda(t)$ 满足：

$$-\dot{\lambda} = \frac{\partial H}{\partial x}, \quad \lambda(T) = \frac{\partial \Phi}{\partial x(T)}$$

梯度：

$$\frac{\partial J}{\partial u} = \frac{\partial H}{\partial u}, \quad \frac{\partial J}{\partial x_0} = \lambda(0)$$

## 离散打靶 / multiple shooting

分段状态/控制变量时，伴随在 segment 边界传递，与 [Trajectory Optimization](../methods/trajectory-optimization.md) 的 multiple shooting 配套。

## 函数光滑化（课程 5.1 关联）

非光滑项（如 $\|x\|_1$、max、碰撞惩罚）常用 Huber、log-barrier、softplus 光滑化后再做伴随，否则次梯度方法或 smoothing 技巧并用。

## 常见误区

- **伴随 = 反向传播**：结构类似，但伴随针对连续/物理约束 OCP，实现细节不同。
- **忽略 checkpoint 内存**：长轨迹需存储或重算中间状态。
- **非光滑处直接 AD**：可能得到错误次梯度，需 smoothing。

## 与其他页面的关系

- [Trajectory Optimization](../methods/trajectory-optimization.md)
- [LQR / iLQR](../methods/lqr-ilqr.md)
- [Optimal Control](../concepts/optimal-control.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*
- [Crocoddyl](../entities/crocoddyl.md) — 高效 iLQR / FDDP 实现

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 5 章 5.1–5.2 光滑化与伴随
