---
type: formalization
tags: [optimization, conic-optimization, friction-cone, trajectory-optimization]
status: complete
updated: 2026-06-23
related:
  - ./friction-cone.md
  - ./quadratic-programming.md
  - ../methods/time-optimal-path-parameterization.md
  - ../methods/penalty-barrier-augmented-lagrangian.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "对称锥规划：将摩擦锥、范数约束与 TOPP 等机器人问题统一为锥优化，并用增广拉格朗日求解。"
---

# Symmetric Cone Programming（对称锥规划）

**对称锥规划**：目标与约束可表示在对称锥（非负正交锥、二阶锥、半正定锥等）上的凸优化问题；比 LP/QP 更一般，能精确刻画摩擦锥、力矩椭球与部分时间最优问题。

## 一句话定义

> 当约束形如 $x \in \mathcal{K}$（$\mathcal{K}$ 为对称锥）时，问题进入锥规划（SOCP / SDP）范畴——比 QP 多表达「锥形约束」，比一般 NLP 更结构化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SOCP | Second-Order Cone Program | 二阶锥规划，$\|Ax+b\|\le c^Tx+d$ |
| SDP | Semidefinite Program | 半定规划，$X\succeq 0$ |
| LP | Linear Programming | 线性规划，非负锥特例 |
| ALM | Augmented Lagrangian Method | 对称锥上常用 PHR 形式 |
| TOPP | Time-Optimal Path Parameterization | 可表述为锥约束的速度规划 |

## 为什么重要

- **精确摩擦锥**：SOC 比 polyhedral 线性化更紧（见 [Friction Cone](./friction-cone.md)）。
- **TOPP / 时间最优**：沿路径的速度曲线常含 $|v'|\le a_{\max}$ 等锥约束。
- **姿态 / 抓取松弛**：SDP 松弛旋转矩阵 $R^TR=I$。

## 核心结构

### 常见对称锥

| 锥 | 形式 | 机器人例子 |
|----|------|-----------|
| 非负正交锥 | $x \ge 0$ | 接触力非负、时间分配 |
| 二阶锥 SOC | $\|(x_1,\ldots,x_{n-1})\|_2 \le x_n$ | 摩擦锥、力矩限制 |
| 半正定锥 | $X \succeq 0$ | 旋转/协方差松弛 |

### 增广拉格朗日（课程 4.2）

对称锥约束 $x \in \mathcal{K}$ 时，PHR 增广拉格朗日将锥投影嵌入迭代：

$$\mathcal{L}_\rho(x,\lambda) = f(x) + \langle \lambda, c(x) \rangle + \frac{\rho}{2} \|\Pi_{\mathcal{K}}(c(x)+\lambda/\rho) - \lambda/\rho\|^2$$

其中 $\Pi_{\mathcal{K}}$ 为锥投影（SOC 投影有闭式）。

## 与 QP 的关系

- SOCP **严格包含** QP（二次约束可转 SOC）
- 工程 WBC 常用 **QP 近似 SOC** 以换 OSQP 速度
- 离线规划 / TOPP 更倾向保留 SOC 结构

## 常见误区

- **所有锥问题都有实时解**：SDP 规模一大就难实时；SOCP 中等规模可行。
- **锥规划 = 凸**：锥规划指结构；目标/约束还需保持凸性。
- **与 NMPC 混淆**：NMPC 一般非凸 NLP，锥规划是其凸子问题或松弛。

## 与其他页面的关系

- [Time-Optimal Path Parameterization](../methods/time-optimal-path-parameterization.md)
- [Penalty / Barrier / Augmented Lagrangian](../methods/penalty-barrier-augmented-lagrangian.md)
- [Convex Relaxation in Robotics](../methods/convex-relaxation-robotics.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Boyd, *Convex Optimization* — SOCP / SDP 章节
- [Friction Cone](./friction-cone.md)

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 4 章对称锥规划
