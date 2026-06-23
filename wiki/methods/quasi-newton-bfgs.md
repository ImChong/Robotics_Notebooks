---
type: method
tags: [optimization, quasi-newton, bfgs, trajectory-optimization, numerical-methods]
status: complete
updated: 2026-06-23
related:
  - ./line-search-steepest-descent.md
  - ./conjugate-gradient-method.md
  - ../entities/curobo.md
  - ./trajectory-optimization.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "拟牛顿 BFGS / L-BFGS：用低秩更新近似 Hessian 逆，是 cuRobo 等 GPU TrajOpt 与大规模 NLP 的默认选择。"
---

# Quasi-Newton BFGS / L-BFGS（拟牛顿法）

**拟牛顿法**：不显式形成 Hessian，用梯度差分维护近似矩阵 $B_k$ 或 $H_k \approx (\nabla^2 f)^{-1}$。**BFGS** 是最常用更新；**L-BFGS** 只存最近 $m$ 步向量，适合高维 TrajOpt。

## 一句话定义

> 用「上一步梯度变化」猜曲率，比最速下降快、比精确牛顿省——机器人 TrajOpt 工业默认。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFGS | Broyden–Fletcher–Goldfarb–Shanno | 拟 Hessian 逆的秩-2 更新 |
| L-BFGS | Limited-memory BFGS | 只存 $m$ 对 $(s,y)$ 向量 |
| NLP | Nonlinear Programming | TrajOpt 打靶后的标准形式 |
| GN | Gauss-Newton | 最小二乘 TrajOpt 的曲率来源 |
| SQP | Sequential Quadratic Programming | 每步解 QP，与 BFGS 可组合 |

## 核心更新（BFGS）

令 $s_k = x_{k+1}-x_k$，$y_k = \nabla f_{k+1}-\nabla f_k$：

$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T, \quad \rho_k = 1/(y_k^T s_k)$$

每步方向 $p_k = -H_k \nabla f_k$，再 **线搜索** 定步长。

**L-BFGS**：不显式存 $H_k$，用 two-loop recursion 计算 $p_k$，内存 $O(mn)$。

## 机器人中的典型应用

| 场景 | 用法 |
|------|------|
| [cuRobo](../entities/curobo.md) | 并行 L-BFGS + 线搜索，GPU 批量 TrajOpt |
| 运动规划 NLP | 关节路径 + 时间参数联合优化 |
| 离线全身 TrajOpt | 数千维决策变量 |
| IK / retargeting | 非凸最小二乘 |

## 优势与局限

| 优势 | 局限 |
|------|------|
| 超线性收敛（凸光滑） | 非凸时仍陷局部极小 |
| L-BFGS 可扩展至高维 | 需良好初值与代价 scaling |
| 与并行 seed 天然兼容 | 约束问题需 SQP / ALM 外壳 |

## 主要分类

| 类型 | 内存 | 适用 |
|------|------|------|
| BFGS | $O(n^2)$ | 中低维 NLP |
| L-BFGS | $O(mn)$ | 高维 TrajOpt、IK |
| 有限内存 + 并行 seed | GPU 批量 | cuRobo 类 motion gen |

## 与其他页面的关系

- [Convex Functions](../formalizations/convex-functions.md) — 凸光滑时超线性收敛
- [Line Search Steepest Descent](./line-search-steepest-descent.md)
- [Conjugate Gradient Method](./conjugate-gradient-method.md) — 大型线性子问题替代
- [Trajectory Optimization](./trajectory-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 6
- [cuRobo](../entities/curobo.md) — 工程实例

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.2 拟牛顿法
