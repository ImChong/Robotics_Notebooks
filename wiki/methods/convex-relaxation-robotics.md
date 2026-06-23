---
type: method
tags: [optimization, convex-relaxation, non-convex, robotics, gnc]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/convex-functions.md
  - ../concepts/optimal-control.md
  - ./penalty-barrier-augmented-lagrangian.md
  - ../formalizations/riemannian-manifold-tangent-space.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "机器人凸松弛：QCQP 松弛、Riemannian Staircase、分布式松弛与 GNC，用于姿态、抓取、配准等非凸问题的可解近似。"
---

# Convex Relaxation in Robotics（机器人凸松弛）

**凸松弛**：将非凸 QCQP / 组合问题 **放松为凸问题**（SDP/SOCP/LP），求下界或近似解；配合 **Riemannian Staircase**、**分布式松弛** 与 **GNC（Graduated Non-Convexity）** 逐步恢复非凸结构。

## 一句话定义

> 机器人里大量问题本质非凸（旋转、接触模式、对应关系）——凸松弛是「先解一个能算的版本，再收紧」的系统套路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QCQP | Quadratically Constrained Quadratic Program | 二次约束二次规划 |
| SDP | Semidefinite Program | 半定松弛，如 $R^TR=I$ |
| GNC | Graduated Non-Convexity | 逐步加强非凸惩罚 |
| SOCP | Second-Order Cone Program | 范数/摩擦类约束 |
| SVD | Singular Value Decomposition | 投影回可行旋转的常用后处理 |

## 核心方法（课程第 6 章）

### 6.2 QCQP 凸松弛

- 旋转矩阵 $R\in SO(3)$：替换 $R^TR=I$ 为 $XX^T=I$ 的 SDP 松弛
- 双足接触模式：混合整数 → 连续松弛 + 舍入

### 6.3 Riemannian Staircase

在 Stiefel/旋转流形上，从 **低秩** 松弛解逐步 **增加秩**（lift），避免陷入 poor local minima；与 [Riemannian Manifold](../formalizations/riemannian-manifold-tangent-space.md) 几何一致。

### 6.4 分布式凸松弛

多智能体 / 大规模因子图：按子图做局部松弛 + 一致性 ADMM，用于多机器人编队、分布式 SLAM 近似。

### 6.5 GNC（Graduated Non-Convexity）

Black & Rangarajan 框架：非凸代价 $\rho(x)$ 用 **凸 surrogate 族** $\rho(x,\mu)$ 参数化，$\mu$ 从凸逐步收紧到原函数；鲁棒估计、点云配准、感知 outlier 常用。

### 函数光滑化（课程 5.1 关联）

Huber / Geman-McClure 等 **鲁棒核** 可视为 GNC 的连续版本，与 [Adjoint Sensitivity](../formalizations/adjoint-sensitivity-analysis.md) 配合做 TrajOpt。

## 机器人应用

| 问题 | 松弛类型 |
|------|---------|
| 全局姿态 / 旋转同步 | SDP / spectral |
| 抓取力闭合 | LP / SOCP 近似 |
| 点云配准 outlier | GNC / ICP 变体 |
| 足端接触时序 | 混合整数松弛 |

## 常见误区

- **松弛解总是可行**：需 projection / rounding 回物理可行集。
- **SDP 可实时**：规模一大仅离线；在线用 SOCP/QP 近似。
- **GNC 保证全局最优**：仅启发式，依赖初值与 schedule。

## 主要分类

| 方法 | 非凸来源 | 松弛类型 |
|------|---------|---------|
| SDP 旋转松弛 | $R^TR=I$ | 半定 |
| Riemannian Staircase | 低秩 lift | 流形秩递增 |
| GNC | 鲁棒核 / outlier | 逐步收紧凸 surrogate |
| 分布式 ADMM | 因子图 | 局部 SOCP/QP |

## 与其他页面的关系

- [Optimal Control](../concepts/optimal-control.md) — 非凸 OCP 出口
- [Symmetric Cone Programming](../formalizations/symmetric-cone-programming.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Carlone et al., rotation estimation / certifiably correct 系列
- Yang et al., GNC survey

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 5 章 5.1、第 6 章凸松弛
