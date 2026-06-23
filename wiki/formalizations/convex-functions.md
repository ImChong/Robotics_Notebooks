---
type: formalization
tags: [optimization, convex-optimization, numerical-methods, foundational]
status: complete
updated: 2026-06-23
related:
  - ./kkt-conditions.md
  - ./quadratic-programming.md
  - ../concepts/constrained-optimization.md
  - ../entities/numerical-optimization-curriculum.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "凸集与凸函数：机器人 QP/MPC/TrajOpt 中「可解、可证、可热启动」问题的数学基础。"
---

# Convex Functions（凸函数）

**凸函数**：定义域为凸集、且函数图像在任意两点连线之上的函数；凸优化问题的局部最优即全局最优，是 WBC QP、凸 MPC、摩擦锥线性化等工程问题的理论基石。

## 一句话定义

> 若 $f(\alpha x + (1-\alpha)y) \le \alpha f(x) + (1-\alpha)f(y)$ 对所有 $x,y$ 与 $\alpha\in[0,1]$ 成立，则 $f$ 为凸函数；机器人里「二次代价 + 线性/仿射约束」几乎总是凸问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QP | Quadratic Programming | 凸二次目标 + 线性约束的标准形式 |
| SOC | Second-Order Cone | 二阶锥；摩擦锥、范数约束的常见凸表示 |
| KKT | Karush–Kuhn–Tucker | 约束凸问题的一阶最优性条件 |
| NMPC | Nonlinear Model Predictive Control | 非线性滚动优化；一般非凸 |
| PSD | Positive Semi-Definite | 半正定；凸二次项 Hessian 需 $\succeq 0$ |

## 为什么重要

- **QP / 凸 MPC 可解性**：Hessian $\succeq 0$ 时，WBC、质心 MPC 有全局最优且多项式时间可解。
- **稳定性与唯一性**：严格凸 + 可行域有内点时，最优解唯一，热启动有效。
- **非凸问题的判据**：TrajOpt、NMPC 非凸时，需要凸松弛（见 [Convex Relaxation](../methods/convex-relaxation-robotics.md)）或好的初值。

## 核心结构

### 凸集

集合 $C$ 为凸集，当且仅当 $\forall x,y\in C,\ \alpha\in[0,1]:\ \alpha x+(1-\alpha)y\in C$。

机器人常见凸集：
- 线性不等式 $Ax\le b$（关节限位、半空间）
- 二阶锥 $\|Ax+b\|_2 \le c^T x + d$（摩擦锥、力矩范数）
- 仿射子空间 $Ax=b$（动力学等式、接触约束）

### 凸函数判别

可微时：$f$ 凸 $\Leftrightarrow$ $\nabla^2 f(x) \succeq 0$ 处处成立。

**Jensen 不等式（定义等价形式）**：

$$f(\alpha x + (1-\alpha) y) \le \alpha f(x) + (1-\alpha) f(y), \quad \alpha \in [0,1]$$

其中 $x,y$ 在 $f$ 的定义域内，$\alpha$ 为插值系数。

常见凸函数：
- 二次型 $x^T P x + q^T x + r$（$P\succeq 0$）
- 范数 $\|x\|_p$（$p\ge 1$）
- 负对数 barrier $-\log(x)$（定义域 $x>0$）
- 最大函数 $\max_i f_i(x)$（$f_i$ 凸则凸）

### 高阶信息（课程 1.2）

- **梯度** $\nabla f$：一阶下降方向
- **Hessian** $\nabla^2 f$：曲率；牛顿法 / Gauss-Newton 的核心
- **强凸**：$\nabla^2 f \succ 0$ 时收敛更快、条件数更好

## 机器人中的典型用法

| 场景 | 凸结构 |
|------|--------|
| WBC QP | $\min \frac12 z^T H z + g^T z$ s.t. 线性约束 |
| 凸 MPC | 线性化动力学 + 二次跟踪代价 |
| 控制分配 | $\min \|W(\tau - \tau_d)\|^2$ s.t. $B\tau = w$ |
| 碰撞距离（SDF 局部） | 线性化后仍为 QP 子问题 |

## 常见误区

- **「二次就是凸」**：只有 $P\succeq 0$ 时才是；不定二次可能非凸。
- **「摩擦锥直接进 QP」**：精确 SOC 需锥求解器；工程常 polyhedral 线性化（见 [Friction Cone](./friction-cone.md)）。
- **忽略定义域**：$\log$、$\sqrt{\cdot}$ 等只在特定域凸。

## 与其他页面的关系

- 最优性：[KKT Conditions](./kkt-conditions.md)
- 标准形式：[Quadratic Programming](./quadratic-programming.md)
- 非凸出口：[Convex Relaxation in Robotics](../methods/convex-relaxation-robotics.md)
- 课程地图：[Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Boyd & Vandenberghe, *Convex Optimization* Ch 3–4
- [HQP](../concepts/hqp.md) — 凸 QP 在 WBC 中的工程实例

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 1 章 1.2–1.3 凸集与凸函数
