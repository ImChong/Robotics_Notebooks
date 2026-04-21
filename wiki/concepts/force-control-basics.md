---
type: concept
tags: [control, manipulation, hardware, force-control]
status: complete
updated: 2026-04-21
related:
  - ./impedance-control.md
  - ./hybrid-force-position-control.md
  - ../methods/actuator-network.md
sources:
  - ../../sources/papers/contact_dynamics.md
summary: "力控制基础（Force Control Basics）介绍了机器人如何通过直接控制执行器扭矩或利用末端力传感器反馈，实现与环境的柔顺物理交互。"
---

# Force Control Basics (力控制基础)

在人形机器人和操作任务中，**力控制 (Force Control)** 是实现物理交互的基石。与传统工业机器人仅跟踪位置轨迹（Position Control）不同，力控制允许机器人感知并调节它对环境施加的力。

## 核心分类

1. **间接力控 (Indirect Force Control)**：
   - 不直接闭环力信号，而是通过调节位置误差与力之间的动态关系来实现柔顺。
   - 代表：[阻抗控制 (Impedance Control)](./impedance-control.md)。
2. **直接力控 (Direct Force Control)**：
   - 显式地设定目标力 $F_d$，并通过力传感器反馈闭环。
   - 代表：[力位混合控制 (Hybrid Control)](./hybrid-force-position-control.md)。

## 实现手段

- **基于电流的力矩估计**：利用电机的电流-力矩常数 ($K_t$) 估算输出力，成本低但受摩擦影响大。
- **六维力传感器 (F/T Sensor)**：安装在手腕或足端，测量最真实、最精确的接触力。
- **关节力矩传感器**：ANYmal 等高性能机器人使用，测量每个关节的真实输出。

## 为什么难

- **接触不稳定**：由于传感器噪声和总线延迟，高增益的力控极易引起系统震荡（Instability）。
- **非线性摩擦**：减速器的静摩擦会掩盖真实的力反馈。

## 关联页面
- [Impedance Control (阻抗控制)](./impedance-control.md)
- [Hybrid Force-Position Control](./hybrid-force-position-control.md)
- [Actuator Network (执行器网络)](../methods/actuator-network.md)

## 参考来源
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md)
- Siciliano, B. *Robotics: Modelling, Planning and Control*.
