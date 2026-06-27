---
type: method
tags: [optimization, bfgs, quasi-newton, numerical-methods, trajectory-optimization]
status: complete
updated: 2026-06-27
summary: "BFGS 用梯度差分低秩更新近似 Hessian 逆，凸光滑问题超线性收敛，是中低维 NLP 的经典拟牛顿法。"
related:
  - ./l-bfgs.md
  - ./quasi-newton-bfgs.md
  - ./newtons-method.md
  - ./line-search-steepest-descent.md
  - ../formalizations/convex-functions.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
---

# BFGS（Broyden–Fletcher–Goldfarb–Shanno）

**BFGS**：**拟牛顿法** 中最常用的秩-2 更新，维护近似 Hessian 逆 $H_k \approx (\nabla^2 f)^{-1}$，每步方向 $p_k = -H_k \nabla f_k$ 再 [线搜索](./line-search-steepest-descent.md)；在 [凸光滑](../formalizations/convex-functions.md) 问题上 **超线性收敛**，存储 $O(n^2)$。

## 一句话定义

> 不算 Hessian，用「这一步梯度比上一步变了多少」来猜曲率——中等维度 NLP 的拟二阶捷径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFGS | Broyden–Fletcher–Goldfarb–Shanno | 拟 Hessian 逆的秩-2 更新 |
| QN | Quasi-Newton | 拟牛顿法族 |
| NLP | Nonlinear Programming | 无约束/罚化后 NLP |
| GN | Gauss-Newton | 最小二乘的另一种曲率来源 |
| L-BFGS | Limited-memory BFGS | 高维版，见 [l-bfgs.md](./l-bfgs.md) |

## 主要技术路线

### 1. BFGS 更新公式

令 $s_k = x_{k+1}-x_k$，$y_k = \nabla f_{k+1}-\nabla f_k$，$\rho_k = 1/(y_k^T s_k)$：

$$
H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T
$$

满足 **割线方程** $H_{k+1} y_k = s_k$。

### 2. 与牛顿法的关系

- 首步常取 $H_0 = I$ 或对角缩放。
- 收敛速度介于梯度下降与 [Newton](./newtons-method.md) 之间（超线性）。
- 无需显式 Hessian，但 **$O(n^2)$ 内存** 限制维度。

### 3. 何时选 BFGS vs L-BFGS

| 维度 $n$ | 推荐 |
|---------|------|
| $<$ 几千 | 完整 BFGS 可行 |
| 几千～数万 | [L-BFGS](./l-bfgs.md) |
| GPU 批量 TrajOpt | L-BFGS + 并行 seed |

## 优势与局限

| 优势 | 局限 |
|------|------|
| 超线性收敛（凸光滑） | $O(n^2)$ 存储，高维不适用 |
| 实现简单、教科书标准 | 非凸仍陷局部极小 |
| 与线搜索成熟配套 | 约束需 SQP/ALM 外壳 |

## 在机器人中的典型应用

- **中低维 TrajOpt 原型**、离线规划验证
- **IK 小规模问题**（维度不高时）
- **教学与算法对照**：理解 [L-BFGS](./l-bfgs.md) 的 two-loop 推导

## 关联页面

- [L-BFGS](./l-bfgs.md)
- [Quasi-Newton BFGS 总览](./quasi-newton-bfgs.md)
- [Newton's Method](./newtons-method.md)
- [Gauss-Newton](./gauss-newton.md)
- [Line Search & Steepest Descent](./line-search-steepest-descent.md)
- [Convex Functions](../formalizations/convex-functions.md)
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md) — Broyden/Fletcher/Goldfarb/Shanno (1970)
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.2

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 6.1–6.3
- [Broyden (1970)](https://doi.org/10.1090/S0002-9947-1970-0258249-9)
