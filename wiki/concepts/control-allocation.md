---
type: concept
tags: [control, optimization, overactuated, allocation, humanoid]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/quadratic-programming.md
  - ../concepts/whole-body-control.md
  - ../concepts/constrained-optimization.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "控制分配：冗余驱动系统（多旋翼、推力矢量、多接触）在等式动力学约束下求可行且最优的执行器指令。"
---

# Control Allocation（控制分配）

**控制分配**：给定低维 **广义力/ wrench 指令** $w \in \mathbb{R}^m$（如期望合外力/力矩），在冗余执行器映射 $w = B\tau$ 下求执行器输入 $\tau$，并满足饱和、速率与效率等约束——典型 **凸 QP** 或 **最小范数** 问题。

## 一句话定义

> 上层控制器说「需要这样的合力/力矩」，控制分配回答「各电机/螺旋桨各出多少力」——多旋翼、卫星姿态、冗余机械臂都用到。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QP | Quadratic Programming | 带框约束的加权最小二乘分配 |
| WBC | Whole-Body Control | 全身控制含更广义的力矩分配 |
| NMPC | Nonlinear Model Predictive Control | 可在 MPC 层联合优化分配 |
| DoF | Degrees of Freedom | 执行器数常大于 wrench 维数 |
| SDF | Signed Distance Field | 与碰撞优化不同子问题 |

## 问题形式

**等式**：$B\tau = w$（$B$ 为 effectiveness matrix）

**常见目标**：
$$\min_\tau \ \|W(\tau - \tau_{\text{pref}})\|^2 \quad \text{s.t.} \quad B\tau = w,\ \tau_{\min} \le \tau \le \tau_{\max}$$

无可行解时引入 **slack** 或 **层级 QP**（先最小化 $\|B\tau-w\|$，再优化 $\|\tau\|$）。

## 机器人实例

| 平台 | $w$ | $\tau$ |
|------|-----|--------|
| 多旋翼 | 4D wrench（推力+力矩） | 各旋翼转速/推力 |
| 人形双足 | 接触 wrench | 各接触点力 |
| 冗余机械臂 | 末端 6D wrench | 关节力矩 |
| 推力矢量飞行器 | 体轴力/力矩 | 各发动机推力+偏转 |

## 常见误区

- **与 WBC 混淆**：WBC 含全身动力学与任务堆叠；控制分配常指 **actuator 层** 的 $B\tau=w$。
- **忽略饱和**：未处理 infeasible 时指令会被静默截断，导致跟踪误差。
- **忽略效率权重 $W$**：均匀分配未必能耗最优。

## 与其他页面的关系

- [Quadratic Programming](../formalizations/quadratic-programming.md)
- [Whole-Body Control](./whole-body-control.md)
- [Constrained Optimization](./constrained-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- Johansen & Fossen, control allocation 综述
- [Multirotor Simulation Stack](../overview/multirotor-simulation-planning-control-stack.md)

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 3 章 3.6 控制分配
