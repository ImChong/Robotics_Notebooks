---
type: method
tags: [optimization, conjugate-gradient, numerical-methods, linear-systems]
status: complete
updated: 2026-06-27
related:
  - ./quasi-newton-bfgs.md
  - ../formalizations/quadratic-programming.md
  - ../entities/numerical-optimization-curriculum.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "共轭梯度法：求解大型稀疏 SPD 线性系统与严格凸 QP 子问题，内存低于直接分解。"
---

# Conjugate Gradient Method（共轭梯度法）

**共轭梯度（CG）**：在 $n$ 维空间构造 $A$-共轭方向，至多 $n$ 步解精确线性系统 $Ax=b$（精确算术）；对大型稀疏 SPD 矩阵，截断 CG 是求解凸 QP 牛顿步、弹性力学与 PDE 约束优化的主力。

## 一句话定义

> 不用显式存 $A^{-1}$，用共轭方向迭代解 $Ax=b$ —— 大规模严格凸 QP 子问题的经典内核。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CG | Conjugate Gradient | 共轭梯度迭代 |
| PCG | Preconditioned CG | 预条件 CG，改善条件数 |
| SPD | Symmetric Positive Definite | 对称正定，CG 适用条件 |
| QP | Quadratic Programming | 牛顿步即解 SPD 系统 |
| KKT | Karush–Kuhn–Tucker | 约化 KKT 系统可 SPD 化 |

## 核心思想

对 SPD 的 $A$，选搜索方向 $p_k$ 满足 $p_i^T A p_j = 0$（$i\ne j$），一维最优步长有闭式：

$$\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}, \quad r_k = b - A x_k$$

**预条件** $M \approx A$：解 $M^{-1}Ax = M^{-1}b$ 加速收敛。

## 机器人中的用法

- **严格凸 QP 内层**：牛顿法每步解 KKT 的 reduced system
- **弹性 / 软约束仿真** 中的大型线性子问题
- **与 L-BFGS 分工**：CG 解线性子问题，L-BFGS 解非线性外层

## 优势与局限

| 优势 | 局限 |
|------|------|
| 内存 $O(n)$ | 仅适用于 SPD（或适当变换） |
| 稀疏结构友好 | 病态时需预条件 |
| 可 early stop 近似解 | 非凸 NLP 外层仍需其他方法 |

## 主要分类

| 变体 | 特点 |
|------|------|
| 标准 CG | SPD 系统精确解（$n$ 步） |
| PCG | 预条件加速 |
| 截断 CG | 近似牛顿步，early stop |

## 与其他页面的关系

- [Quadratic Programming](../formalizations/quadratic-programming.md)
- [Quasi-Newton BFGS](./quasi-newton-bfgs.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 5
- Golub & Van Loan, *Matrix Computations*

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.3 共轭梯度
