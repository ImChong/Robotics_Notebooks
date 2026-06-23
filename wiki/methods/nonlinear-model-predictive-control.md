---
type: method
tags: [control, mpc, nmpc, optimization, nonlinear-programming]
status: complete
updated: 2026-06-23
related:
  - ./model-predictive-control.md
  - ../concepts/constrained-optimization.md
  - ../methods/penalty-barrier-augmented-lagrangian.md
  - ../queries/mpc-solver-selection.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "非线性模型预测控制 NMPC：每步求解非线性 OCP，可处理完整动力学与避障，是足式/人形高级控制的常用框架。"
---

# Nonlinear Model Predictive Control（NMPC）

**NMPC（非线性 MPC）**：在每个控制周期求解 **非线性** 有限时域 OCP——非线性动力学 $x_{k+1}=f(x_k,u_k)$、非线性代价与约束（碰撞、摩擦等），只执行首步控制后滚动重规划。

## 一句话定义

> 线性 MPC 的「完全体」：不线性化掉动力学，直接解 NLP——更强表达力，更高算力需求。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NMPC | Nonlinear Model Predictive Control | 非线性滚动时域优化 |
| MPC | Model Predictive Control | 广义的滚动 OCP；常指线性/凸特例 |
| OCP | Optimal Control Problem | NMPC 每步的数学对象 |
| SQP | Sequential Quadratic Programming | NMPC 常用求解框架 |
| RTI | Real-Time Iteration | 每周期只做有限 SQP 迭代 |

## 与线性/凸 MPC 的区别

| 维度 | 凸/线性 MPC | NMPC |
|------|------------|------|
| 动力学 | 线性化固定 | 完整非线性 |
| 求解 | QP（OSQP 等） | NLP（Acados SQP-RTI、IPOPT） |
| 实时性 | 50–500 Hz 常见 | 10–100 Hz，依赖 horizon |
| 约束 | 线性/凸 | 任意光滑约束 |

## 典型求解流程

1. **打靶 / 多重打靶** 离散 OCP
2. **SQP-RTI** 或 **iLQR**：每周期 warm start 上一步解
3. **增广拉格朗日 / 内点** 处理等式动力学与不等式
4. 执行 $u_0^*$，移位初值，重复

## 机器人应用（课程 3.8–3.9）

- 足式 / 人形 **全身 NMPC**（CD-MPC、Whole-body NMPC 等研究线）
- **避障 loco-manipulation**：碰撞距离约束进 NLP
- 与 WBC 分层：NMPC 出 wrench/CoM 参考，WBC 做 QP 跟踪

## 优势与局限

| 优势 | 局限 |
|------|------|
| 完整动力学与约束 | 非凸、初值敏感 |
| 统一处理避障与任务 | 需严格实时预算与 fallback |
| 与 TrajOpt 共享工具链 | 调参（horizon、权重）成本高 |

## 主要分类

| 层级 | 方法 | 频率 |
|------|------|------|
| 凸 NMPC 近似 | 线性化 + QP | 50–500 Hz |
| SQP-RTI | 有限 SQP 步 | 20–100 Hz |
| 完整 NLP | IPOPT / 多步 SQP | 离线 / 慢环 |

## 与其他页面的关系

- [Model Predictive Control](./model-predictive-control.md) — 线性/凸基础
- [Trajectory Optimization](./trajectory-optimization.md) — 离线同源 OCP
- [MPC Solver Selection](../queries/mpc-solver-selection.md)
- [Constrained Optimization](../concepts/constrained-optimization.md)

## 推荐继续阅读

- [Humanoid Motion Control Know-How](../queries/humanoid-motion-control-know-how.md) — SRBD vs CD-NMPC 路线
- Acados / OCS2 文档

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章 3.8–3.9 NMPC
