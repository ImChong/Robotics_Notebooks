---
type: concept
tags: [friction, actuator, system-identification, sim2real, modeling]
status: complete
updated: 2026-07-09
related:
  - ./system-identification.md
  - ./friction-compensation.md
  - ./robot-link-and-rotor-inertia.md
  - ../entities/bam-better-actuator-models.md
  - ../entities/paper-bam-extended-friction-servo-actuators.md
  - ../methods/actuator-network.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "关节摩擦常用 Coulomb + Viscous + Stribeck 组合建模；四足 SysID 中摩擦参数对 Sim2Real 跟踪误差与跌倒率影响显著。"
---

# Joint Friction Models（关节摩擦模型）

**关节摩擦模型** 描述传动与轴承中 **与速度、负载相关的非线性阻力**，是 URDF 默认参数往往缺失、却强烈影响 **力矩跟踪与 Sim2Real** 的关键项。

## 一句话定义

> 电机输出的 $\tau_{\text{cmd}}$ 不等于关节净力矩——**摩擦项** 吃掉一部分，建模不准则仿真策略在真机上「打滑」或振荡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Coulomb | Coulomb Friction | 与速度符号相关的恒定摩擦 |
| Viscous | Viscous Friction | 与角速度成正比的阻尼项 |
| Stribeck | Stribeck Effect | 低速区摩擦峰值后随速度下降 |
| SysID | System Identification | 摩擦参数常靠辨识获得 |
| PD | Proportional–Derivative | 摩擦未补偿时 PD 需更大增益 |
| DR | Domain Randomization | 可对摩擦分布随机化 |
| BAM | Better Actuator Models | 扩展摩擦+伺服执行器建模方向 |

## 为什么重要

课程 Ch3 将摩擦与 **206 个 URDF 参数** 并列：四足膝部 **转子惯量可达 link 百倍**，摩擦在低速换向时同样显著。Project 3 对比 **无补偿 / 摩擦补偿 / 补偿+DR** 三组实验。

## 常用模型

### 1. Coulomb + Viscous

$$
\tau_f = \tau_c \,\mathrm{sign}(\dot{q}) + b\,\dot{q}
$$

- $\tau_c$：库仑摩擦矩
- $b$：粘性系数

简单、可辨识，但 **零速附近不连续**。

### 2. Stribeck（库仑 + 粘性 + 低速峰）

$$
\tau_f = \bigl(\tau_c + (\tau_s - \tau_c)\,e^{-|\dot{q}/\dot{q}_s|^\alpha}\bigr)\mathrm{sign}(\dot{q}) + b\,\dot{q}
$$

- $\tau_s$：静摩擦峰值；$\dot{q}_s$：Stribeck 速度尺度

更贴近 **换向瞬间** 的真实行为，SysID 参数更多。

### 3. 扩展执行器模型

见 [BAM](../entities/bam-better-actuator-models.md)：摩擦 + 齿槽 + 伺服带宽联合建模。

## 辨识与可微拟合

| 方法 | 思路 |
|------|------|
| 经典回归 / 最小二乘 | 正弦扫频实验，拟合 $\tau_c, b$ |
| 可微仿真 + 梯度 | 仿真轨迹 MSE 对摩擦参数求导（课程 `jax.grad`） |
| 执行器网络 | 数据驱动补偿残差（[Actuator Network](../methods/actuator-network.md)） |

## 常见误区

- **只在仿真里调 PD，不建模摩擦**：真机低速抖动、站立漂移。
- **Stribeck 参数过拟合单条轨迹**：换负载或温度后失效，需 DR 覆盖。

## 关联页面

- [Friction Compensation](./friction-compensation.md)
- [System Identification](./system-identification.md)
- [Sim2Real](./sim2real.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)
- [仿真物理保真度链路](../queries/simulation-physics-fidelity.md) — 摩擦模型属第 ③ 接触/摩擦层
- [Physics Fidelity ↔ Sim2Real Gap](./physics-fidelity-sim2real-gap.md) — 摩擦简化如何转化为打滑/接触 gap

## 推荐继续阅读

- Bledt et al., *Extended friction models for servo actuators* — 见 [paper-bam](../entities/paper-bam-extended-friction-servo-actuators.md)
- [WBC Implementation Guide](../queries/wbc-implementation-guide.md) — 摩擦补偿代码片段

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch3 摩擦模型
