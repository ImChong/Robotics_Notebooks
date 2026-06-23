---
type: method
tags: [control, pid, classical-control, joint-control, quadruped]
status: complete
updated: 2026-06-23
related:
  - ./model-predictive-control.md
  - ./reinforcement-learning.md
  - ../comparisons/mpc-vs-rl.md
  - ../entities/quadruped-control-curriculum.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "PID 是关节级经典反馈控制：比例-积分-微分组合跟踪期望位置/速度；四足演进路线中作为底层执行层，复杂地形 loco 常由 RL 取代高层。"
---

# PID Control（比例-积分-微分控制）

**PID 控制**：根据误差 $e$ 及其积分、微分，输出控制量 $u$ 的经典 **线性反馈** 方法，仍是机器人 **关节伺服、姿态稳定** 的默认底层。

## 一句话定义

> 用 **现在误差（P）、历史累积（I）、变化趋势（D）** 三部分加权，把输出拉回设定值。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PID | Proportional–Integral–Derivative | 比例-积分-微分控制 |
| PD | Proportional–Derivative | 足式 loco 常用（省略 I 防 windup） |
| MPC | Model Predictive Control | 预测优化，PID 之后的模型驱动升级 |
| RL | Reinforcement Learning | 数据驱动高层，常仍用 PD 执行 |
| FF | Feedforward | 重力/摩擦前馈与 PID 并联 |
| SDK | Software Development Kit | 实机底层常暴露 PD 增益接口 |

## 在四足控制演进中的位置

课程 Ch1 叙事：

```
PID（单关节伺服）
  → MPC（质心/足端轨迹优化，需模型）
    → RL（复杂地形端到端，PD 仍作执行层）
```

- **PID/PD**：适合 **已知轨迹跟踪**、桌面级调试、站立平衡初版
- **局限**：多关节耦合、接触切换、非结构化地形下 **手工调参难扩展**（MPC 在碎石地也会失效，从而引出 RL）

## 离散形式（位置 PD）

$$
\tau = K_p (q^* - q) + K_d (\dot{q}^* - \dot{q}) + \tau_{\text{ff}}
$$

四足 RL 部署典型栈：**RL 输出 $q^*, \dot{q}^*$ 或 $\tau$** → PD 转力矩 → 电机。

## 整定要点（衔接 Sim2Real）

- **增益过高**：噪声放大、振荡、过热
- **增益过低**：跟踪慢、抗扰差、跌倒
- 课程 Ch6 强调 **实机 PD 整定与安全协议** 与摩擦补偿、DR 并列

## 主要技术路线

### 1. 独立关节 PID

每关节独立 $K_p, K_i, K_d$，适合 **单轴标定、桌面臂**；足式多关节耦合时增益互相牵制。

### 2. 并联前馈（重力 / 摩擦）

$$
\tau = K_p e + K_d \dot{e} + \tau_{\text{gravity}}(q) + \hat{\tau}_f(\dot{q})
$$

- 重力项常由 [RNEA](../formalizations/articulated-body-algorithms.md) 计算
- 摩擦项见 [Joint Friction Models](../concepts/joint-friction-models.md)、[Friction Compensation](../concepts/friction-compensation.md)

### 3. RL + PD 执行层（四足主流）

RL 输出目标位置/力矩，**PD 仍作底层伺服**；复杂地形 loco 由 [PPO](./ppo.md) 等承担，PID 不消失而是 **退居执行层**（见 [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)）。

### 4. 向 MPC / RL 演进

当单关节 PID 无法处理接触切换与非凸地形，升级到 [MPC](./model-predictive-control.md)（模型预测）或 [RL](./reinforcement-learning.md)（数据驱动）；选型见 [MPC vs RL](../comparisons/mpc-vs-rl.md)。

## 常见误区

- **在四足复杂地形上纯 PID 做 loco**：缺少接触调度与步态，难以与 RL/MPC 路线竞争。
- **积分项盲目开启**：饱和 windup，足式更常用 **PD + 前馈**。

## 关联页面

- [Model Predictive Control](./model-predictive-control.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [MPC vs RL](../comparisons/mpc-vs-rl.md)
- [Friction Compensation](../concepts/friction-compensation.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)

## 推荐继续阅读

- Åström & Murray, *Feedback Systems*
- [Humanoid RL Cookbook](../queries/humanoid-rl-cookbook.md) — PD 增益与部署

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch1 控制演进
