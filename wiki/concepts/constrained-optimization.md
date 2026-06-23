---
type: concept
tags: [optimization, constrained-optimization, mpc, wbc, numerical-methods]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/kkt-conditions.md
  - ../formalizations/quadratic-programming.md
  - ../methods/penalty-barrier-augmented-lagrangian.md
  - ../methods/nonlinear-model-predictive-control.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "约束优化：机器人动力学、接触、碰撞与安全集必须以等式/不等式进入优化；本页梳理形式分类与复杂度直觉。"
---

# Constrained Optimization（约束优化）

**约束优化**：在等式 $h(x)=0$ 与不等式 $g(x)\le 0$ 下最小化目标 $f(x)$；机器人 OCP、WBC、NMPC 的本质都是约束优化——动力学是等式，摩擦/碰撞/输入饱和是不等式。

## 一句话定义

> 机器人「能做什么」由约束定义；「做得多好」由目标定义——控制就是解这个约束优化问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCP | Optimal Control Problem | 动力学等式 + 状态/输入约束 |
| QP | Quadratic Programming | 线性约束 + 凸二次目标 |
| NLP | Nonlinear Programming | 一般非线性约束优化 |
| LP | Linear Programming | 线性目标与约束 |
| NMPC | Nonlinear Model Predictive Control | 滚动时域 NLP |

## 形式分类与复杂度（课程 3.1）

| 类型 | 结构 | 典型算法 | 机器人例子 |
|------|------|---------|-----------|
| LP | 线性目标 + 线性约束 | 单纯形 / 内点 | 低维几何、资源分配 |
| QP | 凸二次 + 线性约束 | OSQP / active-set | WBC、凸 MPC |
| SOCP | 锥约束 | 锥内点 / ALM | 摩擦锥、TOPP |
| NLP | 非线性约束 | SQP / iLQR / IPM | 全身 TrajOpt、NMPC |
| MINLP | 含整数 | 分支定界 | 接触模式离散（混合） |

**低维特例**（课程 3.2–3.3）：2D/3D 约束多边形内 LP/QP 可有 **Seidel 类线性时间** 算法，用于几何规划子模块。

## 三种处理约束的思路

1. **直接法**：SQP、内点法，每步解凸子问题（见 [KKT](../formalizations/kkt-conditions.md)）
2. **序列无约束化**：罚函数、障碍法（见 [Penalty / Barrier / ALM](../methods/penalty-barrier-augmented-lagrangian.md)）
3. **松弛 / 投影**：凸松弛（见 [Convex Relaxation](../methods/convex-relaxation-robotics.md)）、CBF 安全滤波

## 机器人应用索引

- 控制分配 → [Control Allocation](./control-allocation.md)
- 碰撞距离 → [Collision Distance Optimization](./collision-distance-optimization.md)
- NMPC → [Nonlinear MPC](../methods/nonlinear-model-predictive-control.md)

## 常见误区

- **约束越多越好**：过紧导致 infeasible，WBC 常见。
- **忽略可行性**：NMPC 需 fallback（软约束 /  previous solution）。
- **LP/QP/NLP 混用术语**：结构决定算法，先分类再选型。

## 与其他页面的关系

- [Optimal Control](../concepts/optimal-control.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)
- [MPC Solver Selection](../queries/mpc-solver-selection.md)

## 推荐继续阅读

- [Constrained Optimization 课程地图](../entities/numerical-optimization-curriculum.md)
- Boyd, *Convex Optimization*

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章约束优化
