---
type: method
tags: [optimization, motion-planning, time-optimal, trajectory-optimization, manipulation]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/symmetric-cone-programming.md
  - ./trajectory-optimization.md
  - ../entities/curobo.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "时间最优路径参数化 TOPP：几何路径固定，优化沿路径的速度曲线以满足关节/加速度极限并最小化时间。"
---

# Time-Optimal Path Parameterization（TOPP）

**TOPP（时间最优路径参数化）**：给定关节空间或任务空间 **几何路径** $q(s)$（$s$ 为路径参数），求速度 $\dot{s}(t)$ 或 $\dot{s}(s)$ 曲线，使 **总时间最小** 且满足各关节速度/加速度/torque 限幅——常表述为 **凸 SOCP 或 LP** 子问题。

## 一句话定义

> 路径形状已定，只问「沿这条路多快跑」——机械臂 pick-and-place 与 AGV 时间标定的经典步骤。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TOPP | Time-Optimal Path Parameterization | 沿路径的时间最优速度规划 |
| TOPP-RA | TOPP with Reachability Analysis | 考虑动力学可达性的变体 |
| SOCP | Second-Order Cone Program | TOPP 常见凸形式 |
| OCP | Optimal Control Problem | 完整 TrajOpt 含路径+时间联合 |
| FK | Forward Kinematics | 路径常先在关节或笛卡尔空间给定 |

## 核心结构

路径参数化：$q(t) = q(s(t))$，则 $\dot{q} = q'(s)\dot{s}$，$\ddot{q} = q''(s)\dot{s}^2 + q'(s)\ddot{s}$。

约束 $\|\dot{q}\|_\infty \le v_{\max}$、$\|\ddot{q}\|_\infty \le a_{\max}$ 可转为 **$\dot{s}$、$\ddot{s}$ 的线性或锥约束**。

目标：$\min T = \int_0^S \frac{1}{\dot{s}(s)} ds$ 或离散时间等价形式。

## 与 TrajOpt 的关系

| TOPP | 全轨迹优化 |
|------|-----------|
| 路径固定，只优化时间律 | 路径与时间联合 NLP |
| 常凸，毫秒级 | 非凸，更慢 |
| 工业运动控制后处理 | 研究 / 高难度动作 |

## 机器人中的用法

- 机械臂 **time-scaling**（MoveIt / 控制器后处理）
- 移动机器人沿 Dubins / Hermite 路径的时间最优
- 课程 4.3：对称锥规划应用入口

## 常见误区

- **忽略 jerk 限制**：加速度可行但 jerk 不可执行。
- **路径几何不佳**：TOPP 无法修复自碰撞路径。
- **与 TOP（task-space time-optimal policy）混淆**：本页指路径参数化，非 RL 策略名。

## 主要分类

| 表述 | 求解 |
|------|------|
| LP 时间参数化 | 线性规划 |
| SOCP 速度/加速度锥 | 锥规划 / TOPP-RA |
| NLP 联合路径+时间 | 全 TrajOpt |

## 与其他页面的关系

- [Symmetric Cone Programming](../formalizations/symmetric-cone-programming.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Pham et al., TOPP-RA 系列
- Modern Robotics Ch 9 时间缩放（简化版）

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 4 章 4.3 TOPP
