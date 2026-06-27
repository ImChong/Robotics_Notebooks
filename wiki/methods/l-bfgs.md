---
type: method
tags: [optimization, l-bfgs, quasi-newton, trajectory-optimization, numerical-methods]
status: complete
updated: 2026-06-27
summary: "L-BFGS 只存最近 m 对梯度差分向量，用 two-loop recursion 计算搜索方向，是高维 TrajOpt 与 cuRobo 类运动生成的工业默认。"
related:
  - ./bfgs.md
  - ./quasi-newton-bfgs.md
  - ./gauss-newton.md
  - ../entities/curobo.md
  - ./trajectory-optimization.md
  - ../formalizations/convex-functions.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
---

# L-BFGS（Limited-memory BFGS）

**L-BFGS**：[BFGS](./bfgs.md) 的 **有限内存** 变体，只保留最近 $m$ 对 $(s_k, y_k)$，用 **two-loop recursion** 隐式计算 $H_k \nabla f_k$，内存 $O(mn)$；是 [cuRobo](../entities/curobo.md) 等 GPU 批量 [Trajectory Optimization](./trajectory-optimization.md) 的默认下降引擎。

## 一句话定义

> 只记住最近几步的梯度变化来近似曲率——高维 TrajOpt 上「用得起的二阶信息」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| L-BFGS | Limited-memory BFGS | Liu & Nocedal 1989 |
| BFGS | Broyden–Fletcher–Goldfarb–Shanno | 满内存版拟牛顿 |
| TrajOpt | Trajectory Optimization | 典型高维应用场景 |
| NLP | Nonlinear Programming | 打靶后的非线性规划 |
| m | History Size | 存储对数，典型 5–20 |

## 主要技术路线

### 1. Two-loop recursion

不显式存 $H_k$，由 $\{s_i, y_i\}_{i=k-m}^{k-1}$ 递归计算 $q = H_k g_k$：

1. **后向循环**：用 $s_i, y_i$ 修正 $q$
2. **前向循环**：对称修正得到搜索方向

每步 $O(mn)$ 时间、$O(mn)$ 内存。

### 2. 与 Gauss-Newton 的分工

| 问题结构 | 常见选择 |
|---------|---------|
| 显式残差 $r(x)$，可算 $J$ | [Gauss-Newton](./gauss-newton.md) / LM |
| 一般 NLP / 已罚化约束 | L-BFGS + 线搜索 |
| 批量并行 seed | GPU L-BFGS（cuRobo） |

### 3. 实践要点

- **历史长度 $m$**：增大 $m$ 改善曲率估计但增内存；$m=10$ 是常见默认。
- **缩放**：首步对角 $H_0$ 缩放（Nocedal 建议 $\gamma_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}$）。
- **线搜索**：仍需 Armijo/Wolfe 保证下降。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 可扩展至 $10^4$+ 维 | 非凸仍依赖初值与 seed |
| 与 GPU 批量天然兼容 | 约束需外层 ALM/SQP |
| 工业 TrajOpt 成熟默认 | 不如结构化 iLQR 利用动力学 |

## 在机器人中的典型应用

- **[cuRobo](../entities/curobo.md)**：并行 L-BFGS 运动生成
- **全身 TrajOpt**、操作空间路径优化
- **IK / retargeting** 高维非凸最小化
- **离线规划**：多初值 + L-BFGS 取最优

## 关联页面

- [BFGS](./bfgs.md)
- [Quasi-Newton BFGS 总览](./quasi-newton-bfgs.md)
- [Gauss-Newton](./gauss-newton.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [cuRobo](../entities/curobo.md)
- [Convex Functions](../formalizations/convex-functions.md)
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md) — Liu & Nocedal (1989)
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.2

## 推荐继续阅读

- [Liu & Nocedal (1989)](https://doi.org/10.1007/BF01582236)
- [cuRobo](../entities/curobo.md) — 工程实例
