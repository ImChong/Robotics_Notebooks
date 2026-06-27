---
type: method
tags: [optimization, quasi-newton, bfgs, trajectory-optimization, numerical-methods]
status: complete
updated: 2026-06-27
related:
  - ./bfgs.md
  - ./l-bfgs.md
  - ./line-search-steepest-descent.md
  - ./conjugate-gradient-method.md
  - ./newtons-method.md
  - ../entities/curobo.md
  - ./trajectory-optimization.md
  - ../comparisons/second-order-optimizers.md
sources:
  - ../../sources/papers/second_order_optimizers.md
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "拟牛顿 BFGS / L-BFGS 总览：分别见独立节点 bfgs.md 与 l-bfgs.md；是 cuRobo 等 GPU TrajOpt 的默认选择。"
---

# Quasi-Newton BFGS / L-BFGS（拟牛顿法总览）

**拟牛顿法**：不显式形成 Hessian，用梯度差分维护近似矩阵。本页为 **总览入口**；各算法细节见独立节点 **[BFGS](./bfgs.md)** 与 **[L-BFGS](./l-bfgs.md)**。

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

## 核心更新（详见独立节点）

- **[BFGS](./bfgs.md)**：满内存秩-2 更新，$O(n^2)$ 存储，中低维 NLP。
- **[L-BFGS](./l-bfgs.md)**：two-loop recursion，$O(mn)$ 内存，高维 TrajOpt 工业默认。

令 $s_k = x_{k+1}-x_k$，$y_k = \nabla f_{k+1}-\nabla f_k$，BFGS 更新公式见 [bfgs.md](./bfgs.md)。

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

- [BFGS](./bfgs.md) · [L-BFGS](./l-bfgs.md) — 独立算法节点
- [Second-Order Optimizers 对比](../comparisons/second-order-optimizers.md)
- [Newton's Method](./newtons-method.md) · [Gauss-Newton](./gauss-newton.md)
- [Convex Functions](../formalizations/convex-functions.md) — 凸光滑时超线性收敛
- [Line Search Steepest Descent](./line-search-steepest-descent.md)
- [Conjugate Gradient Method](./conjugate-gradient-method.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 6
- [cuRobo](../entities/curobo.md) — 工程实例

## 参考来源

- [Second-Order Optimizers 论文摘录](../../sources/papers/second_order_optimizers.md)
- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.2 拟牛顿法
