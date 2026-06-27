---
type: method
tags: [optimization, newton, second-order, numerical-methods, trajectory-optimization]
status: complete
updated: 2026-06-27
summary: "牛顿法用 Hessian 构造局部二次模型求搜索方向，在强凸邻域二次收敛，是理解 Gauss-Newton、LM 与截断牛顿的曲率基准。"
related:
  - ./gauss-newton.md
  - ./levenberg-marquardt.md
  - ./truncated-newton.md
  - ./line-search-steepest-descent.md
  - ../formalizations/convex-functions.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
---

# Newton's Method（牛顿法）

**牛顿法（Newton's method）**：在当前点用 **Hessian** $\nabla^2 f(x_k)$ 构造二次模型，搜索方向 $p_k = -(\nabla^2 f(x_k))^{-1}\nabla f(x_k)$；在 [凸函数](../formalizations/convex-functions.md) 强凸邻域内 **二次收敛**，是二阶优化的理论基准。

## 一句话定义

> 用曲率矩阵修正梯度方向——在「碗底」附近一步跳得很准，在远处可能乱跳。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Newton | Newton's Method | 二阶局部二次模型迭代 |
| Hessian | Hessian Matrix | $\nabla^2 f$，曲率信息 |
| NLP | Nonlinear Programming | 无约束/约束非线性规划 |
| LM | Levenberg-Marquardt | 阻尼牛顿在最小二乘上的变体 |
| GN | Gauss-Newton | 最小二乘的 Hessian 近似 |

## 主要技术路线

### 1. 基本更新

$$
p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k), \qquad x_{k+1} = x_k + \alpha_k p_k
$$

$\alpha_k$ 由 [线搜索](./line-search-steepest-descent.md) 确定，保证全局收敛性。

### 2. 收敛性质

- **局部二次收敛**：初始点足够接近强凸极小值且 Hessian Lipschitz 时，误差平方衰减。
- **全局问题**：非凸时 Hessian 可能不定，需 **修正**（特征值裁剪、信赖域）或转 [Levenberg-Marquardt](./levenberg-marquardt.md)。

### 3. 与机器人问题的关系

| 场景 | 用法 |
|------|------|
| 小规模 NLP | 直接形成并分解 Hessian |
| 最小二乘 TrajOpt | 常用 [Gauss-Newton](./gauss-newton.md) 替代精确 Hessian |
| 大规模问题 | [Truncated Newton](./truncated-newton.md) + Hessian-vector product |
| iLQR forward pass | 结构化牛顿方向 + 线搜索 |

### 4. 工程难点

- 形成 Hessian：$O(n^2)$ 存储，$O(n^3)$ 分解。
- 非凸不定：需修正或信赖域外壳。
- 高维 TrajOpt：很少用「全牛顿」，多用 GN / L-BFGS / 截断牛顿。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 强凸邻域收敛极快 | Hessian 贵且可能不定 |
| 理论清晰，是二阶方法母型 | 高维直接牛顿不可行 |
| 与 QP 子问题、KKT 系统一体 | 需良好初值 |

## 在机器人中的典型应用

- **NMPC / TrajOpt 原型**：低维打靶问题验证代价 landscape。
- **状态估计非线性最小二乘**：常被 GN/LM 替代。
- **iLQR**：利用动力学结构做 **结构化牛顿**，而非显式密集 Hessian。

## 关联页面

- [Gauss-Newton](./gauss-newton.md)
- [Levenberg-Marquardt](./levenberg-marquardt.md)
- [Truncated Newton](./truncated-newton.md)
- [Line Search & Steepest Descent](./line-search-steepest-descent.md)
- [Convex Functions](../formalizations/convex-functions.md)
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)
- [Trajectory Optimization](./trajectory-optimization.md)

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md) — Dennis & Schnabel / Kantorovich 现代表述
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 1 章 1.5 修正阻尼牛顿法

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 6.1–6.3
- Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization* Ch 6
