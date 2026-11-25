---
type: method
tags: [optimization, convex-relaxation, non-convex, robotics, gnc]
status: complete
updated: 2026-08-09
related:
  - ../formalizations/convex-functions.md
  - ../concepts/optimal-control.md
  - ./penalty-barrier-augmented-lagrangian.md
  - ../formalizations/riemannian-manifold-tangent-space.md
original: ../../sources/papers/graesdal_tight_convex_relaxations_arxiv_2402_10312.md
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

## 为什么重要

- **不依赖初值的近全局规划（Graesdal et al., RSS 2024 arXiv:2402.10312v2）**：含旋转几何和接触开关的多体规划问题高度非凸，传统直接轨迹优化极易陷入局部极小值；通过半定规划（SDP）对接触模态内 QCQP 做紧松弛并组合凸集图（GCS），可在无人工初值下求出接近全局最优的推挤接触序列。

## 核心原理（核心方法）

```mermaid
flowchart LR
  NC["非凸机器人问题<br/>(SO(3) 旋转 / 接触开关 / 离群点)"] --> CR["构造凸松弛或近似族<br/>(QCQP->SDP / Riemannian Staircase / GNC)"]
  CR --> SOL["全局/下界求解或逐步收紧"]
  SOL --> PR["SVD 投影 / 局部 NLP 舍入"]
  PR --> FE["物理可行姿态 / 接触序列执行"]
```

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

## 工程实践

- **SDP 松弛求解 + 廉价局部舍入（Rounding）**：离线先解带秩-1 约束的紧凸松弛下界（或用对角/低阶锥逼近降耗时），再执行轻量局部 NLP 舍入并下发混合 MPC 真机跟踪（Kuka LBR iiwa 7 验证）；目前大型 SDP 尚不适于毫秒级在线反推。

## 常见误区或局限

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

## 关联页面

- [Optimal Control](../concepts/optimal-control.md) — 非凸 OCP 出口
- [Symmetric Cone Programming](../formalizations/symmetric-cone-programming.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Carlone et al., rotation estimation / certifiably correct 系列
- Yang et al., GNC survey

## 参考来源

- [graesdal_tight_convex_relaxations_arxiv_2402_10312.md](../../sources/papers/graesdal_tight_convex_relaxations_arxiv_2402_10312.md) — Graesdal et al., *Towards Tight Convex Relaxations for Contact-Rich Manipulation* (RSS 2024, arXiv:2402.10312v2)
- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 5 章 5.1、第 6 章凸松弛
