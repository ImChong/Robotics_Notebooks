---
type: formalization
tags: [optimization, constrained-optimization, kkt, numerical-methods]
status: complete
updated: 2026-06-23
related:
  - ./convex-functions.md
  - ./quadratic-programming.md
  - ../concepts/constrained-optimization.md
  - ../methods/penalty-barrier-augmented-lagrangian.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "KKT 条件：约束优化问题的一阶最优性条件，是理解 QP 对偶、NMPC 拉格朗日乘子与 WBC 约束活跃集的基础。"
---

# KKT Conditions（KKT 条件）

**KKT（Karush–Kuhn–Tucker）条件**：带不等式与等式约束的优化问题在最优点的必要条件（凸问题且 Slater 成立时为充分条件），将「找最优解」转化为「解代数方程组 + 互补松弛」。

## 一句话定义

> 在约束优化最优点，梯度可被约束梯度的线性组合「平衡」，且不等式约束要么不紧、要么对应乘子为正——这就是 KKT。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| KKT | Karush–Kuhn–Tucker | 约束优化一阶最优性条件 |
| QP | Quadratic Programming | KKT 退化为线性方程组 + 互补条件 |
| ALM | Augmented Lagrangian Method | 用 KKT 结构构造可微罚项 |
| NMPC | Nonlinear Model Predictive Control | 每步 NLP 的 KKT 点即候选最优 |
| LICQ | Linear Independence Constraint Qualification | 约束梯度线性无关的正则性条件 |

## 为什么重要

- **QP 求解器内核**：OSQP、qpOASES 的 active-set / ADMM 都在追踪 KKT 结构。
- **WBC 无解诊断**：infeasible 时检查哪些等式/不等式冲突。
- **NMPC / TrajOpt**：内点法、SQP、增广拉格朗日都在迭代逼近 KKT 点。

## 标准形式与 KKT

$$\min_x f(x) \quad \text{s.t.} \quad g_i(x) \le 0,\ h_j(x) = 0$$

引入 Lagrangian：

$$\mathcal{L}(x,\lambda,\nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)$$

**KKT 条件**（必要，凸+Slater 下充分）：

1. **平稳性**：$\nabla_x \mathcal{L} = 0$
2. **原始可行**：$g_i(x)\le 0,\ h_j(x)=0$
3. **对偶可行**：$\lambda_i \ge 0$
4. **互补松弛**：$\lambda_i g_i(x) = 0$

## QP 特例

凸 QP：

$$\min_x \frac12 x^T P x + q^T x \quad \text{s.t.} \ Ax \le b,\ Cx = d$$

其中 $P$ 为 Hessian（半正定时问题凸），$q$ 为线性项系数，$x$ 为决策变量。

KKT 为线性系统 + 互补条件；active-set 方法通过猜测活跃约束集 $A$ 解 KKT 线性方程。

## 机器人实例

| 问题 | KKT 含义 |
|------|---------|
| WBC QP | 力矩在最优时平衡任务梯度与约束法向 |
| 控制分配 | 等式 $B\tau=w$ 的 $\nu$；框约束 $\tau_{\min}\le\tau\le\tau_{\max}$ 的 $\lambda$ |
| NMPC | 动力学等式约束的协态变量（见 [Adjoint](../formalizations/adjoint-sensitivity-analysis.md)） |

## 常见误区

- **KKT 点不一定是全局最优**（非凸 NLP）。
- **互补松弛不等于约束 inactive**：等式约束无互补项。
- **忽略 LICQ / Slater**：病态约束下 KKT 可能不适用。

## 与其他页面的关系

- [Constrained Optimization](../concepts/constrained-optimization.md)
- [Quadratic Programming](./quadratic-programming.md)
- [Penalty / Barrier / Augmented Lagrangian](../methods/penalty-barrier-augmented-lagrangian.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 12
- [WBC Implementation Guide](../queries/wbc-implementation-guide.md) — QP 无解时的工程处理

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章 3.5 KKT 与 PHR 增广拉格朗日
