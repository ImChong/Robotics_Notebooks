---
type: formalization
tags: [control, whole-body-control, math, optimization, dynamics]
status: complete
updated: 2026-04-21
related:
  - ../concepts/whole-body-control.md
  - ../concepts/floating-base-dynamics.md
  - ./friction-cone.md
sources:
  - ../../sources/papers/whole_body_control.md
summary: "任务空间逆动力学（TSID）形式化：描述了如何在浮动基座动力学约束下，将多个并行的任务空间加速度映射为一致的关节力矩输出，是 WBC 最主流的数学实现方式。"
---

# Task Space Inverse Dynamics (TSID) 形式化

**TSID** 是一种在保持机器人物理一致性的前提下，实现多任务并行控制的数学框架。它将复杂的运动指令转换为底层的电机力矩。

## 数学定义：QP 形式

TSID 通常被形式化为一个**带等式与不等式约束的二次规划 (QP)** 问题。

### 1. 决策变量
$$ \mathcal{X} = [\ddot{q}^T, \mathbf{f}_{ext}^T, \tau^T]^T $$
其中 $\ddot{q}$ 是关节加速度，$\mathbf{f}_{ext}$ 是接触力，$\tau$ 是力矩。

### 2. 硬约束 (Hard Constraints)
- **浮动基座动力学**：
  $$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = S^T \tau + \sum J_i^T \mathbf{f}_{ext, i} $$
- **接触一致性**（支撑脚不滑移）：
  $$ J_i \ddot{q} + \dot{J}_i \dot{q} = 0 $$
- **摩擦锥约束**：详见 [Friction Cone](./friction-cone.md)。

### 3. 目标函数 (Cost Function)
$$ \min_{\mathcal{X}} \sum w_k \| J_k \ddot{q} + \dot{J}_k \dot{q} - \ddot{x}_{des, k} \|^2 + \lambda \|\tau\|^2 $$
其中 $\ddot{x}_{des, k}$ 是各个任务（如 Base 平衡、手部跟踪）在任务空间期望的加速度。

## 为什么 TSID 强大

- **自动协调**：你只需告诉手去哪里，TSID 会自动计算为了补偿手部动量，躯干应该如何反向摆动。
- **优先级处理**：通过不同的权重 $w_k$，确保安全任务优先于动作性能。

## 关联页面
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)
- [Friction Cone 形式化](./friction-cone.md)

## 参考来源
- Prete, A., et al. (2016). *Task Space Inverse Dynamics*.
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)
