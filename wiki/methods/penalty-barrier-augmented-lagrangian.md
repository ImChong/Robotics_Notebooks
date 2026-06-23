---
type: method
tags: [optimization, penalty-method, barrier-method, augmented-lagrangian, constrained-optimization]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/kkt-conditions.md
  - ../formalizations/symmetric-cone-programming.md
  - ../concepts/constrained-optimization.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "罚函数、障碍法与 PHR 增广拉格朗日：将约束优化转为序列无约束或易求解子问题的三类经典方法。"
---

# Penalty / Barrier / Augmented Lagrangian（罚函数、障碍法与增广拉格朗日）

**序列无约束化**（课程 3.4）：通过罚项或障碍项把约束「推入」目标，反复求解无约束/简单约束子问题。**PHR 增广拉格朗日**（Powell–Hestenes–Rockafellar）在对偶空间更新乘子，收敛性优于纯外点罚函数。

## 一句话定义

> 不想每步解硬约束 KKT？就把违反约束写进代价，或用增广拉格朗日把等式「拉紧」——NMPC 与锥规划常用。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ALM | Augmented Lagrangian Method | 增广拉格朗日法 |
| PHR | Powell–Hestenes–Rockafellar | 一种 ALM 形式，含乘子更新 |
| IPM | Interior-Point Method | 内点法，与障碍法同源 |
| SQP | Sequential Quadratic Programming | 与 ALM 可视为对偶视角 |
| KKT | Karush–Kuhn–Tucker | ALM 迭代逼近 KKT 点 |

## 三类方法对比

| 方法 | 子问题 | 特点 |
|------|--------|------|
| **外点罚** | $\min f + \mu \|h\|^2 + \mu \|\max(g,0)\|^2$ | 实现简单；$\mu\to\infty$ 病态 |
| **障碍（内点）** | $\min f - \mu \sum \log(-g_i)$ | 保持严格可行；需内点初值 |
| **增广拉格朗日 PHR** | $\min f + \lambda^T h + \frac{\rho}{2}\|h\|^2$ + … | 乘子更新，收敛快；对称锥有锥投影版 |

### PHR 等式约束示意

$$\mathcal{L}_\rho(x,\lambda) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2$$

交替：固定 $(\lambda,\rho)$ 最小化 $\mathcal{L}_\rho$；更新 $\lambda \leftarrow \lambda + \rho h(x^*)$。

对称锥版本见 [Symmetric Cone Programming](../formalizations/symmetric-cone-programming.md)。

## 机器人中的用法

- **NMPC**：Acados / CasADi 常用 SQP-RTI 或 ALM 处理动力学等式
- **接触 / 碰撞软约束**：外点罚或 Huber 罚（课程 5.1 光滑化）
- **TOPP / SOCP**：锥 ALM（课程 4.2、4.4 实践）

## 常见误区

- **罚系数一次设太大**：Hessian 病态，线搜索失败。
- **障碍法从不可行点启动**：直接报错。
- **ALM 不更新乘子**：退化成纯罚函数，收敛慢。

## 主要分类

| 方法 | 机制 | 典型用途 |
|------|------|---------|
| 外点罚 | $\mu\|h\|^2$ 增大 | 软约束 TrajOpt |
| 障碍法 | $-\log(-g)$ 内点 | 严格可行路径 |
| PHR ALM | 乘子 + 增广项 | NMPC、对称锥 |

## 与其他页面的关系

- [KKT Conditions](../formalizations/kkt-conditions.md)
- [Nonlinear MPC](./nonlinear-model-predictive-control.md)
- [Convex Relaxation in Robotics](./convex-relaxation-robotics.md) — GNC 与罚函数哲学不同

## 推荐继续阅读

- Nocedal & Wright, *Numerical Optimization* Ch 17–18
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章 3.4–3.5、第 4 章 4.2
