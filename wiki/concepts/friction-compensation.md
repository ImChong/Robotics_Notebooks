---
type: concept
tags: [friction, control, sim2real, feedforward, quadruped]
status: complete
updated: 2026-08-13
related:
  - joint-module-no-load-velocity-vs-torque-loop.md
  - ./joint-friction-models.md
  - ./sim2real.md
  - ./system-identification.md
  - ../methods/joint-actuator-parameter-identification.md
  - ../methods/ppo.md
  - ../entities/quadruped-control-curriculum.md
  - ../queries/wbc-implementation-guide.md
  - ./gravity-compensation.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
original:
  - ../../sources/papers/bam_extended_friction_reading_arxiv_2410_08650_fc7.md
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
| $\tau_{\text{gravity}}$ | [重力补偿](./gravity-compensation.md)（RNEA） |
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

## 核心原理

补偿的前提是摩擦模型足够准。Duclusaud 等（BAM，arXiv 2410.08650）指出主流仿真器的 Coulomb–Viscous 模型 $\tau_f=-K_v\dot\theta-\mathrm{sign}(\dot\theta)K_c$ 在 $\dot\theta=0$ 处为零且换向不连续，并把摩擦改写为"阻止运动的力矩预算"$\tau_f^m$（实际摩擦被限幅在 $[-\tau_f^m,\tau_f^m]$）。其扩展族 M2–M6 依次加入 Stribeck 效应、负载相关项、方向性与二次项；摩擦随负载 $|\tau_m-\tau_e|$ 增大，是上面"典型实现"中仅含 $\tau_c,b$ 的两参数式所缺的部分。

## 工程实践

- 参数来源：摆锤台架上以 CMA-ES 最小化仿真/实测轨迹 MAE（75% 辨识、25% 验证），含"抬起释放"轨迹以区分反电动势与粘滞（BAM §V）。
- 模型选择不是越复杂越好：Dynamixel 上 M4 在 2R 臂最优，M5 辨识集更好但 2R 臂无提升（作者推测过拟合）；谐波减速器 eRob80:100 上 M6 最优（BAM §VI）。
- 推断：把 BAM 辨识出的 $\hat\tau_f$ 作实机前馈项时，需与低增益 PD 下的力矩饱和、限幅一并考虑；论文本身只验证了仿真侧精度。

## 局限

- BAM 未建模温度、径向力与驻留时间（dwell time），补偿模型随温升漂移时需重新辨识。
- 补偿只缩小系统性偏差，残余误差仍需 DR 等手段兜底（见上文）。
- 以上定量结论来自两类舵机/谐波执行器的台架实验，是否外推到其他关节模组需自行验证。

## 常见误区

- **补偿模型错符号**：换向瞬间力矩加倍振荡。
- **只靠补偿不做安全限幅**：前馈饱和可导致过流；需与 [实机安全协议](../concepts/sim2real.md) 联用。

## 关联页面
- [关节模组空载：速度环与力矩环的输出扭矩区别](joint-module-no-load-velocity-vs-torque-loop.md)

- [Gravity Compensation](./gravity-compensation.md) — 真机示教几乎总是 $g(q)+\hat\tau_f$
- [Joint Friction Models](./joint-friction-models.md)
- [System Identification](./system-identification.md)
- [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) — 补偿项 $\hat{\tau}_f$ 的参数从哪来
- [Privileged Training](./privileged-training.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)

## 推荐继续阅读

- [Sim2Real Checklist](../queries/sim2real-checklist.md)
- [WBC Implementation Guide](../queries/wbc-implementation-guide.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch6 与 Project 3
- [sources/papers/bam_extended_friction_reading_arxiv_2410_08650_fc7.md](../../sources/papers/bam_extended_friction_reading_arxiv_2410_08650_fc7.md) — BAM 扩展摩擦模型全文通读归档
