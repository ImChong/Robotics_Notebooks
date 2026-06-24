---
type: query
tags: [optimization, software, solvers, mpc, wbc, trajectory-optimization]
status: complete
updated: 2026-06-23
related:
  - ./mpc-solver-selection.md
  - ../entities/curobo.md
  - ../entities/crocoddyl.md
  - ../entities/drake.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "机器人优化软件怎么选：QP / NLP / 锥规划求解器与建模框架（CasADi、Drake、Acados、OSQP 等）对照。"
---

# Optimization Software Selection（优化软件选型）

> **Query 产物**：机器人数值优化课程第 5.4 节「优化软件资源」的结构化选型指南；与 [MPC Solver Selection](./mpc-solver-selection.md) 互补（本页覆盖更广 NLP/TrajOpt/建模层）。

## 一句话结论

**先按问题结构选型，再按实时/部署约束选实现**：凸 QP → OSQP/qpOASES/HPIPM；非线性 OCP/NMPC → Acados/OCS2/CasADi+IPOPT；离线 TrajOpt → Crocoddyl/iLQR；GPU 运动生成 → cuRobo；多体+符号 → Drake。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QP | Quadratic Programming | OSQP / qpOASES / HPIPM |
| NLP | Nonlinear Programming | IPOPT / SNOPT / Acados |
| AD | Automatic Differentiation | CasADi / JAX / autograd |
| SQP | Sequential Quadratic Programming | Acados / WORHP 内核 |
| SOCP | Second-Order Cone Program | Mosek / ECOS / Clarabel |

## 决策树（简版）

```
问题类型？
├─ 凸 QP（WBC / 凸 MPC）
│   ├─ 嵌入式 + 热启动 → qpOASES / HPIPM
│   └─ 稀疏 + Python → OSQP
├─ 非线性 OCP / NMPC
│   ├─ 实时 SQP-RTI → Acados / OCS2
│   └─ 原型 / 离线 → CasADi + IPOPT
├─ 离线 TrajOpt（DDP/iLQR）
│   └─ Crocoddyl / Drake TrajOpt
├─ GPU 碰撞 + TrajOpt
│   └─ cuRobo
└─ 锥规划 / SDP
    └─ Mosek / Clarabel / CVXPY
```

## 工具对照表

| 工具 | 强项 | 典型机器人场景 |
|------|------|---------------|
| **OSQP** | 稀疏凸 QP，Python/C | WBC、凸 MPC |
| **qpOASES** | Active-set，热启动 | 嵌入式 WBC |
| **HPIPM** | 结构化 MPC QP | Acados/OCS2 后端 |
| **Acados** | 代码生成 SQP-RTI | NMPC 足式/臂 |
| **CasADi** | 符号 AD + NLP 接口 | 快速原型、算法研究 |
| **Crocoddyl** | FDDP/iLQR | 离线全身 TrajOpt |
| **Drake** | 多体 + 优化 + 仿真 | 研究、系统集成 |
| **cuRobo** | GPU SDF + L-BFGS | 机械臂 motion gen |
| **IPOPT** | 通用大规模 NLP | 离线 NMPC 验证 |
| **CVXPY** | 凸建模 | 松弛/验证下界 |

## 线性系统求解器（课程 5.3）

| 类型 | 方法 | 何时用 |
|------|------|--------|
| 稠密直接 | LU/Cholesky | 小规模 WBC |
| 稀疏直接 | LDL / SuperLU | 大规模 KKT |
| 迭代 | CG / MINRES + 预条件 | 大型 SPD 子问题（见 [CG](../methods/conjugate-gradient-method.md)） |

## 常见误区

- **用 IPOPT 跑 1kHz WBC**：过度；换 OSQP。
- **不生成导数**：NMPC 有限差分太慢，用 CasADi/Drake AD。
- **忽视代码生成**：Acados 相对 CasADi 纯解释执行快一个数量级。

## 与其他页面的关系

- [MPC Solver Selection](./mpc-solver-selection.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)
- [WBC Implementation Guide](./wbc-implementation-guide.md)

## 关联页面

- [Numerical Optimization Method Selection](./numerical-optimization-method-selection.md) — 先选算法（本页选求解器/实现）
- [MPC Solver Selection](./mpc-solver-selection.md) — 凸 MPC / WBC 专项
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md) — 课程软件章节
- [Quadratic Programming](../formalizations/quadratic-programming.md) — QP 求解器对照
- [Nonlinear MPC](../methods/nonlinear-model-predictive-control.md) — Acados / CasADi 场景
- [Trajectory Optimization](../methods/trajectory-optimization.md) — Crocoddyl / IPOPT 场景

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 5 章 5.3–5.4
- OSQP / Acados / Crocoddyl 官方文档
