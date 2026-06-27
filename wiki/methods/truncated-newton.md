---
type: method
tags: [optimization, truncated-newton, newton-cg, numerical-methods, large-scale]
status: complete
updated: 2026-06-27
summary: "截断牛顿法用共轭梯度近似求解 Newton 方程，在 Hessian-vector product 可得时适合大规模稀疏 NLP，是精确牛顿与拟牛顿之间的折中。"
related:
  - ./newtons-method.md
  - ./conjugate-gradient-method.md
  - ./l-bfgs.md
  - ../formalizations/convex-functions.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
---

# Truncated Newton（截断牛顿 / Newton-CG）

**截断牛顿（Truncated Newton）**：每步不求解精确 Newton 方程 $Hp = -g$，而用 [共轭梯度（CG）](./conjugate-gradient-method.md) **迭代近似** 解，达到预设残差阈值即 **early stop**；当 Hessian 稀疏或 **Hessian-vector product** 廉价时，适合大规模无约束 NLP 的 [二阶曲率利用](../formalizations/convex-functions.md)。

## 一句话定义

> 牛顿步太贵就只解个大概——用 CG 迭代凑够好的搜索方向就停。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Truncated Newton | Truncated Newton Method | 不精确牛顿步 |
| Newton-CG | Newton Conjugate Gradient | CG 解 Newton 方程 |
| CG | Conjugate Gradient | 内层线性迭代 |
| HVP | Hessian-Vector Product | $\nabla^2 f \cdot v$，可自动微分 |
| NLP | Nonlinear Programming | 大规模无约束/罚化问题 |

## 主要技术路线

### 1. 内外层结构

**外层**：牛顿迭代 $x_{k+1} = x_k + \alpha_k p_k$

**内层**：用 CG 解 $(\nabla^2 f(x_k)) p = -\nabla f(x_k)$，在 $i$ 步后停止当 $\|r_i\| \le \eta_k \|\nabla f\|$。

### 2. Hessian-vector product

不显式形成 $\nabla^2 f$，用一次反向模式 AD 计算 $Hp$（与深度学习 [反向传播](../concepts/backpropagation.md) 同族），内存 $O(n)$。

### 3. 与 L-BFGS / 精确牛顿对比

| 方法 | 曲率信息 | 内存 | 适用 |
|------|---------|------|------|
| 精确 Newton | 完整 Hessian | $O(n^2)$ | 低维 |
| Truncated Newton | HVP + CG | $O(n)$ | 大规模稀疏/可微 |
| L-BFGS | 梯度历史 | $O(mn)$ | 高维 TrajOpt 默认 |

Nash (2000) 综述指出：截断牛顿在 **部分大规模凸/近凸** 问题上可优于 L-BFGS，但机器人 TrajOpt 工程上 L-BFGS 更常见。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 避免显式 Hessian 矩阵 | 内层 CG 需 SPD 或修正 |
| 与自动微分 HVP 契合 | 实现与调参比 L-BFGS 复杂 |
| 理论二次收敛（精确内层） | 非凸不定 Hessian 需信赖域 |

## 在机器人中的典型应用

- **可微仿真 / 可微 TrajOpt** 原型（HVP 可得时）
- **严格凸 QP 子问题** 已用 CG，截断牛顿用于外层非线性
- **研究型大规模 NLP**；工业 motion gen 仍多 [L-BFGS](./l-bfgs.md)

## 关联页面

- [Newton's Method](./newtons-method.md)
- [Conjugate Gradient Method](./conjugate-gradient-method.md)
- [L-BFGS](./l-bfgs.md)
- [BFGS](./bfgs.md)
- [Convex Functions](../formalizations/convex-functions.md)
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md) — Nash (2000)
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 CG 与牛顿步

## 推荐继续阅读

- [Nash, A Survey of Truncated-Newton Methods (2000)](https://doi.org/10.1137/S0036144500377774)
- Nocedal & Wright, *Numerical Optimization* Ch 7.1–7.2
