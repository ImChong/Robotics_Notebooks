---
type: concept
tags: [friction, control, sim2real, feedforward, quadruped]
status: complete
updated: 2026-06-23
related:
  - ./joint-friction-models.md
  - ./sim2real.md
  - ./system-identification.md
  - ../methods/ppo.md
  - ../entities/quadruped-control-curriculum.md
  - ../queries/wbc-implementation-guide.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "Friction Compensation 在控制力矩上叠加摩擦前馈项，缩小仿真—真机力矩跟踪 gap；四足 Project 3 对比无补偿、补偿与补偿+DR。"
---

# Friction Compensation（摩擦补偿）

**摩擦补偿**：根据关节速度（及可选负载）估计摩擦矩 $\hat{\tau}_f$，在 **前馈通道** 叠加到 PD/RL 输出，使 **净关节力矩** 更接近仿真假设。

## 一句话定义

> $\tau_{\text{out}} = \tau_{\text{policy}} + \hat{\tau}_f(\dot{q})$ —— 用模型或辨识表「抵消」摩擦，而不是全靠 PD 硬扛。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FF | Feedforward | 前馈补偿通道 |
| PD | Proportional–Derivative | 底层力矩/位置跟踪 |
| SysID | System Identification | 摩擦参数来源 |
| DR | Domain Randomization | 与补偿互补的鲁棒化手段 |
| Sim2Real | Simulation to Real | 补偿直接针对力矩跟踪 gap |
| RL | Reinforcement Learning | 策略输出通常仍经 PD + 补偿 |
| SDK | Software Development Kit | 实机底层力矩接口部署点 |

## 为什么重要

课程 Ch6 将摩擦补偿列为 Sim2Real **三大工程手段** 之一（另：RMA 蒸馏、PD 增益整定）。Project 3 量化 **关节跟踪误差** 与 **跌倒率** 三组对比。

## 典型实现

```
τ_cmd = τ_RL + τ_PD + τ_gravity + τ_friction(q̇)
```

| 组件 | 说明 |
|------|------|
| $\tau_{\text{RL}}$ | 策略网络输出（位置/力矩目标） |
| $\tau_{\text{PD}}$ | 跟踪误差反馈 |
| $\tau_{\text{gravity}}$ | 重力补偿（RNEA） |
| $\tau_{\text{friction}}$ | 基于 [Joint Friction Models](./joint-friction-models.md) |

摩擦项例：

$$
\hat{\tau}_f = \tau_c \,\mathrm{sign}(\dot{q}) + b\,\dot{q}
$$

参数来自 SysID 或厂商标定。

## 与 DR / 蒸馏的关系

| 手段 | 作用 |
|------|------|
| 摩擦补偿 | 缩小 **系统性** 力矩偏差 |
| DR | 让策略对 **残余参数误差** 鲁棒 |
| Teacher–Student | 部署时去掉仿真特权，保留运动技能 |

三者 **可叠加**；课程实验设计为「补偿+DR」优于单独一项。

## 常见误区

- **补偿模型错符号**：换向瞬间力矩加倍振荡。
- **只靠补偿不做安全限幅**：前馈饱和可导致过流；需与 [实机安全协议](../concepts/sim2real.md) 联用。

## 关联页面

- [Joint Friction Models](./joint-friction-models.md)
- [System Identification](./system-identification.md)
- [Privileged Training](./privileged-training.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)

## 推荐继续阅读

- [Sim2Real Checklist](../queries/sim2real-checklist.md)
- [WBC Implementation Guide](../queries/wbc-implementation-guide.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch6 与 Project 3
