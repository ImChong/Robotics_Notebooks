---
type: method
tags: [optimization, gauss-newton, least-squares, trajectory-optimization, numerical-methods]
status: complete
updated: 2026-06-27
summary: "Gauss-Newton 对残差最小二乘用 Jacobian 外积 JᵀJ 近似 Hessian，是 TrajOpt 打靶、IK 与标定的默认二阶曲率模型。"
related:
  - ./newtons-method.md
  - ./levenberg-marquardt.md
  - ./trajectory-optimization.md
  - ../formalizations/convex-functions.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
---

# Gauss-Newton（高斯-牛顿法）

**Gauss-Newton（GN）**：最小化 $\|r(x)\|^2$ 时，用残差 Jacobian $J(x) = \partial r / \partial x$ 构造 Hessian 近似 $H \approx J^T J$，搜索方向 $p = -(J^T J)^{-1} J^T r$；避免显式二阶导，是机器人 **非线性最小二乘** 的主力。

## 一句话定义

> 最小二乘里 Hessian ≈ Jacobian 转置乘 Jacobian——用一阶导数信息凑出二阶步长。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GN | Gauss-Newton | 非线性最小二乘的二阶近似 |
| LM | Levenberg-Marquardt | GN + 阻尼，病态时更稳 |
| IK | Inverse Kinematics | 常表述为残差最小化 |
| TrajOpt | Trajectory Optimization | 打靶后多为 GN/L-BFGS |
| J | Jacobian Matrix | 残差对决策变量的雅可比 |

## 主要技术路线

### 1. 问题形式与近似

目标 $f(x) = \frac{1}{2}\|r(x)\|^2$，梯度 $\nabla f = J^T r$。精确 Hessian：

$$
\nabla^2 f = J^T J + \sum_i r_i(x) \nabla^2 r_i(x)
$$

GN **忽略** 二阶残差项，取 $H_{GN} = J^T J$（在 $r$ 小或接近解时合理）。

### 2. 搜索方向

$$
p = -(J^T J)^{-1} J^T r = -\arg\min_p \|J p + r\|^2
$$

即线性化残差的最小二乘步（与一次 [Newton](./newtons-method.md) 步在最小二乘情形下等价）。

### 3. 失效模式与修正

- $J^T J$ **奇异或病态**（冗余参数、欠约束 IK）→ 用 [Levenberg-Marquardt](./levenberg-marquardt.md) 加 $\lambda I$。
- 残差大、远离解 → 二阶项不可忽略，需线搜索或信赖域。

## 优势与局限

| 优势 | 局限 |
|------|------|
| 不需显式 Hessian of $f$ | $J^T J$ 可能奇异/病态 |
| 与 TrajOpt 打靶天然契合 | 远离解时近似差 |
| 可稀疏化、并行计算 $J$ | 大规模需 L-BFGS 或截断 GN |

## 在机器人中的典型应用

- **轨迹打靶 / multiple shooting**：[Trajectory Optimization](./trajectory-optimization.md)
- **IK / 运动重定向**：关节角求逆解
- **传感器标定 / SLAM 束调整**：非线性最小二乘内核
- **状态估计**：非线性最小二乘滤波

## 关联页面

- [Newton's Method](./newtons-method.md)
- [Levenberg-Marquardt](./levenberg-marquardt.md)
- [BFGS](./bfgs.md)
- [L-BFGS](./l-bfgs.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [Convex Functions](../formalizations/convex-functions.md)
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md) — Gauss (1809) / Nocedal & Wright Ch 10
- [数值优化基础课程](../../sources/courses/numerical_optimization_foundations_robotics.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 10.1–10.2
- Madsen, Nielsen & Tingleff, *Methods for Non-Linear Least Squares Problems*
