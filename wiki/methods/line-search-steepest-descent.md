---
type: method
tags: [optimization, numerical-methods, unconstrained-optimization, line-search]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/convex-functions.md
  - ./quasi-newton-bfgs.md
  - ./lqr-ilqr.md
  - ../entities/numerical-optimization-curriculum.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "线搜索最速下降与阻尼牛顿法：非凸无约束 NLP 的基础迭代，也是 iLQR forward pass 线搜索的同族工具。"
---

# Line Search & Steepest Descent（线搜索与最速下降）

**线搜索最速下降**：沿负梯度方向 $-\nabla f(x)$ 搜索步长 $\alpha$ 使目标下降；配合 **Armijo / Wolfe** 条件保证收敛。**修正阻尼牛顿法** 用 Hessian 逆（或正则化）替代梯度方向，二次收敛但需正定修正。

## 一句话定义

> 每步先定方向 $p$，再在一维上找合适步长 $\alpha$ 使 $f(x+\alpha p)$ 足够下降——TrajOpt 里最常见的「下降 + 线搜索」骨架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NLP | Nonlinear Programming | 非线性规划，TrajOpt 离散后即 NLP |
| GD | Gradient Descent | 梯度下降，最速下降特例 |
| GN | Gauss-Newton | 最小二乘的 Hessian 近似，类阻尼牛顿 |
| iLQR | iterative Linear Quadratic Regulator | forward pass 常用线搜索确定步长 |
| LM | Levenberg-Marquardt | Hessian 加阻尼 $\lambda I$ 保证下降 |

## 核心算法

### 最速下降

1. 计算 $g = \nabla f(x)$
2. 方向 $p = -g$
3. 线搜索求 $\alpha$ 满足 Armijo：$f(x+\alpha p) \le f(x) + c_1 \alpha g^T p$
4. 更新 $x \leftarrow x + \alpha p$

收敛慢（病态二次上 zig-zag），但实现简单、每步便宜。

### 修正阻尼牛顿法（课程 1.5）

1. 解 $(H + \lambda I) p = -g$（$H=\nabla^2 f$ 或 GN 近似）
2. 若 $p$ 非下降方向，增大 $\lambda$（Levenberg-Marquardt）
3. 线搜索确定 $\alpha$，更新 $x$

## 机器人中的用法

- **iLQR / DDP forward pass**：应用 $k_k, K_k$ 时需线搜索避免 overshoot
- **小规模 TrajOpt 原型**：Python 手写验证代价 landscape
- **非凸 IK**：梯度 + 线搜索作 baseline，再换 BFGS

## 优势与局限

| 优势 | 局限 |
|------|------|
| 实现简单、每步 $O(n)$ 梯度 | 最速下降收敛慢 |
| 线搜索保证单调下降 | 牛顿法需 Hessian，非凸时可能不定 |
| 与拟牛顿易组合 | 高维直接 Hessian 不可行 |

## 主要分类

| 类型 | 代表 | 适用场景 |
|------|------|---------|
| 一阶 | 最速下降 + Armijo | 原型、高维 cheap 梯度 |
| 二阶 | 阻尼牛顿 / Gauss-Newton | 中小规模、良好初值 |
| 混合 | LM 正则化 Hessian | 非凸最小二乘、IK |

## 与其他页面的关系

- [Convex Functions](../formalizations/convex-functions.md) — 凸情形下降方向保证
- [Quasi-Newton BFGS](./quasi-newton-bfgs.md) — 替代精确 Hessian
- [LQR / iLQR](./lqr-ilqr.md) — 结构化牛顿方向
- [Trajectory Optimization](./trajectory-optimization.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 2–3
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md) — 1.6 实践作业

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 1 章 1.4–1.6
