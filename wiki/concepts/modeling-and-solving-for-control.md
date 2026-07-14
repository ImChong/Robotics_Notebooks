---
type: concept
tags: [control, modeling, optimization, humanoid]
status: complete
updated: 2026-07-14
summary: "飞书「建模+求解」框架：先把人形控制问题写成动力学模型与约束，再选择 OCP/QP/MPC/RL 等求解器；是 Model-based 与 Learning-based 的共同上游。"
related:
  - ./optimal-control.md
  - ./constrained-optimization.md
  - ../formalizations/quadratic-programming.md
  - ./floating-base-dynamics.md
  - ../overview/humanoid-model-based-control-stack.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# 建模与求解（控制问题框架）

飞书 Know-How「**建模 + 求解**」是人形控制问题拆解的第一轴：**建模**确定状态、输入、动力学与约束；**求解**选择能否实时、能否保证约束的算法（QP、MPC、iLQR、RL 等）。

## 一句话定义

先写清「系统怎么动、什么不能违反」，再选「用什么数学工具算控制量」。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCP | Optimal Control Problem | 建模后的标准优化形式 |
| QP | Quadratic Programming | 线性约束二次代价，WBC 常用 |
| MPC | Model Predictive Control | 滚动求解 OCP |
| RL | Reinforcement Learning | 弱显式模型、数据驱动求解 |
| WBC | Whole-Body Control | 建模为约束 QP 的典型实例 |
| MDP | Markov Decision Process | RL 建模框架 |

## 为什么重要

- **避免跳步**：未建模直接调 PPO/MPC 参数常失败。
- **飞书双路线共同前提**：传统路线显式建模；RL 路线至少在仿真里隐含模型。
- **与 Sim2Real 串联**：模型误差是 sim2real 根源之一。

## 核心原理

飞书全文用**四轮小车**四步类比人形控制（见 [建模与求解](../concepts/modeling-and-solving-for-control.md)）：

1. 确定状态 $X$ 与控制输入 $U$（人形：质心 7 维 + 关节 29 维；输入为关节力矩/位置）。
2. 建立动力学 $X_{t+1}=F(X_t,U_t)$：传统用浮动基方程；RL 用 `env.step()` 隐式替代。
3. 构造 OCP：目标（如跟踪 $V_{cmd}$）+ 等式/不等式约束（摩擦、触地速度、力矩限）。
4. 选择求解器：MPC/LQR/DDP（传统）或 MDP+RL（分布意义下的 OCP）。

**Gap 来源：** 传统来自模型简化与参数误差；RL 来自仿真接触/碰撞简化（spring-damper vs time-stepping、隐式/显式积分等）。

## 工程实践

- 用 [Pinocchio](../entities/pinocchio.md) 从 URDF 生成动力学项。
- 手写最小 QP（CVXPY）理解约束激活（飞书传统路线练习）。
- 建模文档与代码 **同一套符号**，避免 README 与实现不一致。

## 局限与风险

- **模型复杂度 vs 实时性**：全刚体精确 vs 简化模型权衡。
- **RL 隐藏建模：** 仿真物理即模型，域随机掩盖误差可能不够。

## 关联页面

- [Optimal Control](./optimal-control.md)
- [Floating-Base Dynamics](./floating-base-dynamics.md)
- [Sim2Real](./sim2real.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [飞书 Know-How 全文](../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md) — §建模+求解
- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [depth-classical-control Stage 0](../../roadmap/depth-classical-control.md)
