---
type: overview
tags: [humanoid, actuator, sea, series-elastic, proprioception, category-hub]
status: complete
updated: 2026-06-02
summary: "Actuator 102 · 05 — SEA 物理退让与储能；双编码器与力矩估计构成「神经系统」；柔顺利安全但增仿真难度。"
related:
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-actuator-102-gear-reflected-inertia.md
  - ./humanoid-actuator-102-decision-species.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_actuator_102.md
  - ../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md
  - ../../sources/papers/humanoid_actuator_102_reference_catalog.md
---

# Actuator 102 · 05：柔顺与感知反馈

> **图谱分类节点**：**VII 柔顺性与串联弹性** + **VIII 感知与反馈**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SEA | Series Elastic Actuator | 串联弹性执行器，提供柔顺与力控 |
| PCB | Printed Circuit Board | 印刷电路板 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |

## 串联弹性（VII）

- **SEA**：电机与负载间弹簧 → 冲击存储、力传感、能效（单腿弹簧效应）。
- 代表路线：**Agility Digit**；理论见 Pratt & Williamson (1995)（[参考文献索引](../../sources/papers/humanoid_actuator_102_reference_catalog.md)）。
- 与刚性谐波/滚柱对比：**带宽受限** 但 **抗摔、抗冲击** 更好。

## 感知（VIII）

- **双编码器**（电机侧 + 关节侧）+ 电流/应变力矩估计。
- 反向驱动能力 **<1 Nm**（理想 <0.5 Nm）否则对地面力「失明」。
- 集成智能关节：驱动 PCB 贴电机、EtherCAT/CAN、热敏电阻 — 对齐 [Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)。

## 仿真权衡

- 刚性执行器：**易建模**，利 RL sim2real（文内 Tesla 隐性理由）。
- SEA/QDD：**弹簧/摩擦/温漂** 使 sim2real 更难。

## 关联页面

- [决策与物种](./humanoid-actuator-102-decision-species.md)
- [参考文献 · SEA / QDD](../../sources/papers/humanoid_actuator_102_reference_catalog.md)

## 参考来源

- [wechat_human_five_humanoid_actuator_102.md](../../sources/blogs/wechat_human_five_humanoid_actuator_102.md)
- [wechat_humanoid_actuator_102_2026-06-02.md](../../sources/raw/wechat_humanoid_actuator_102_2026-06-02.md)
