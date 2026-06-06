---
type: overview
tags: [humanoid, actuator, thermal, torque-control, foc, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 04 — 谐波低效率致热堆积；峰值与持续力矩比目标 3:1+；FOC 电流环 >20kHz、力矩带宽 50–100Hz、指令延迟 <1ms。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-compliance-sensing.md
  - ./humanoid-hardware-101-power-compute-electronics.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
---

# Actuator 102 · 04：热学与力矩控制

> **图谱分类节点**：**V 腿部的热学现实** + **VI 从 PWM 到力矩控制**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| PWM | Pulse-Width Modulation | 脉宽调制，驱动电机与功率器件 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| CAN-FD | CAN with Flexible Data-rate | CAN 的灵活数据速率扩展，更高带宽 |

## 热学（V）

- 谐波 **金属弯曲摩擦** → 连续重载下温升快。
- 行走 **脉冲负载**：支撑相高峰、摆动相近零；但蹲起、持物需 **持续大力矩**。
- **峰值:持续** 目标 **≥3:1**；散热差设计常仅 **~2:1** → 要么峰值不够，要么持续过热。
- **液冷** 可把比例压到 **2:1 甚至 1.5:1**（文内）。

## 控制（VI）

- 从 **PWM 速度环** 升级到 **FOC 电流环 + 力矩控制**。
- 带宽目标（文内 70 kg 膝踝语境）：
  - 电流环 **>20 kHz**
  - 力矩环 **>50–100 Hz**（-3 dB）
  - 指令到力矩 **<1 ms**
- 实现依赖：**低电感**、**48–100 V 母线**、最小背隙/柔顺、**EtherCAT / CAN-FD**。

## 关联页面

- [分离架构](./humanoid-actuator-102-split-architecture.md)
- [决策与物种](./humanoid-actuator-102-decision-species.md)

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
