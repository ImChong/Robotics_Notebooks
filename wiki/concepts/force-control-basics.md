---
type: concept
tags: [control, manipulation, hardware, force-control]
status: complete
updated: 2026-11-22
original:
  - ../../sources/papers/contact_dynamics.md
related:
  - ../queries/contact-wrench-closed-loop.md
  - ./contact-force-loop-bandwidth.md
  - ./impedance-control.md
  - ./hybrid-force-position-control.md
  - ../methods/actuator-network.md
sources:
  - ../../sources/papers/contact_dynamics.md
summary: "力控制基础（Force Control Basics）介绍了机器人如何通过直接控制执行器扭矩或利用末端力传感器反馈，实现与环境的柔顺物理交互。"
---

# Force Control Basics (力控制基础)

## 一句话定义

**力控制（Force Control）** 是通过调节执行器输出力矩或接触位置–力动态映射关系，使机器人在物理接触中自主控制交互力与柔顺性，而非盲目追踪刚性位置轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| F/T | Force/Torque Sensor | 末端六维力/力矩传感器 |
| QDD | Quasi-Direct Drive | 准直驱执行器，依靠低减速比保留高力矩透明度 |
| SEA | Series Elastic Actuator | 串联弹性执行器，将力控转为弹簧形变位置控 |
| QP | Quadratic Programming | 全身力控中求解接触力与力矩分配的标准优化形式 |

## 为什么重要

在人形行走着地与接触丰富操作中，仅靠高增益位置伺服在面对 1–2 mm 环境几何误差时即会触发剧烈碰撞尖峰或卡死；力控制是平衡接触冲击、维持摩擦锥稳定与实现精细装配的底层前提。

## 核心原理

```mermaid
flowchart LR
  CMD["期望力 / 阻抗模型"] --> MODE{"控制架构"}
  MODE -->|直接闭环 F_d| DIR["力反馈 / 力位混合"]
  MODE -->|间接关系 K·Δx + D·ẋ| IND["阻抗 / 导纳控制"]
  DIR --> ACT["执行器实现: Kt 电流环 / F/T / 关节力矩"]
  IND --> ACT
```

1. **间接力控（Indirect Force Control）**：不直接闭环力误差，而是调节位置误差与外力之间的二阶质量–弹簧–阻尼关系，代表为 [阻抗控制（Impedance Control）](./impedance-control.md)。
2. **直接力控（Direct Force Control）**：在选定自由度上显式设定目标力 $F_d$ 并通过传感器反馈闭环，代表为 [力位混合控制（Hybrid Control）](./hybrid-force-position-control.md)。

## 工程实践

- **执行器与传感选型矩阵**：
  - **基于电机相电流 $K_t$**：无需额外力传感器、成本最低，但高频减速器静摩擦会严重掩盖真实外力。
  - **末端六维力传感器（F/T）**：直接测量手腕/足端接触力旋量，精度高但抗冲击与安装重量受限。
  - **关节集成力矩传感器**：如 ANYmal / 协作臂方案，可在全身 QP 力控中实现高频透明闭环。

## 常见误区或局限

- **接触不稳定（Contact Instability）**：传感器噪声、通信延迟与非同位采样（non-collocated sensing）在刚性环境接触时极易诱发高频振荡，高增益直接力控带宽受限于 [力控闭环带宽](./contact-force-loop-bandwidth.md)。
- **非线性静摩擦**：大减速比谐波/行星齿轮的库仑摩擦会吞噬电流估计精度，需要配合摩擦力前馈或 [执行器网络](../methods/actuator-network.md) 补偿。

## 关联页面

- [Query：接触力旋量闭环知识链](../queries/contact-wrench-closed-loop.md) — 本页是四层闭环里的 **② 力旋量表示层**（把接触统一成 6D 力旋量并映回关节）
- [Contact-Force-Loop Bandwidth（力控闭环带宽）](./contact-force-loop-bandwidth.md) — 阻抗/导纳带宽差异的量化边界
- [Impedance Control (阻抗控制)](./impedance-control.md)
- [Hybrid Force-Position Control](./hybrid-force-position-control.md)
- [Actuator Network (执行器网络)](../methods/actuator-network.md)

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md)
- Siciliano, B. *Robotics: Modelling, Planning and Control*.

## 推荐继续阅读

- [Impedance Control](./impedance-control.md) — 阻抗与导纳控制推导
- [Whole-Body Control](./whole-body-control.md) — 多接触点力矩分配的全身 QP 框架
