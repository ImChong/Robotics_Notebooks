---
type: entity
tags: [hardware, actuator, qdd, open-source, vesc, belt-drive]
status: complete
updated: 2026-07-25
related:
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./stanford-doggo-and-pupper.md
  - ./odri-solo-and-bolt.md
  - ./moteus.md
  - ./simplefoc.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/opentorque_actuator.md
  - ../../sources/repos/vesc_bldc.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "OpenTorque Actuator：经典开源准直驱关节（大尺寸外转子航模电机 + 低减速同步带 + VESC + 编码器 + 3D 打印结构），适合快速做 QDD 样机，成熟度与抗冲击不及学术/工业关节。"
---

# OpenTorque Actuator（开源准直驱关节）

## 一句话定义

**OpenTorque Actuator**（[G-Levine/OpenTorque-Actuator](https://github.com/G-Levine/OpenTorque-Actuator)，[Hackaday](https://hackaday.io/project/159404-opentorque-actuator)）是经典 DIY **QDD** 关节：大尺寸外转子航模电机 + 低减速同步带 + **VESC** + 编码器 + 3D 打印结构。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| VESC | Vedder Electronic Speed Controller | 开源大电流 BLDC/FOC 驱动生态 |
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |

## 为什么重要

- 设计叙事接近早期 Cassie / MIT Cheetah 类：**高扭矩密度电机 + 低减速 + 电流力矩控制**。
- 学习阶梯里适合作为 **第一个能转起来的低减速比关节**（在 SimpleFOC/moteus 之后、ODRI 之前）。

## 核心结构/机制

```mermaid
flowchart LR
  mot["大尺寸外转子\n航模 BLDC"]
  belt["同步带\n低减速比"]
  out["关节输出"]
  vesc["VESC FOC"]
  enc["编码器"]
  mot --> belt --> out
  vesc --> mot
  enc --> vesc
```

## 工程实践

- 用它练：电机选型、带传动布置、VESC 电流力矩模式与编码器闭环。
- 再升级到 [ODRI](./odri-solo-and-bolt.md) 学双编码器、专用驱动 PCB 与测试流程。
- VESC 资料见 [vesc_bldc 归档](../../sources/repos/vesc_bldc.md)；注意其 initially 非专为高频关节协议设计。

## 局限与风险

- 体积、重量、轴承布置与抗冲击通常达不到成熟人形要求。
- 仓库许可字段为 `NOASSERTION`，使用前核验 LICENSE。

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [Stanford Doggo](./stanford-doggo-and-pupper.md)（同类同步带 QDD + ODrive）
- [ODRI](./odri-solo-and-bolt.md)
- [moteus](./moteus.md)

## 参考来源

- [sources/repos/opentorque_actuator.md](../../sources/repos/opentorque_actuator.md)
- [sources/repos/vesc_bldc.md](../../sources/repos/vesc_bldc.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)

## 推荐继续阅读

- <https://github.com/G-Levine/OpenTorque-Actuator>
- <https://hackaday.io/project/159404-opentorque-actuator>
