---
type: overview
tags: [topic, topic-communication, ethercat, can, ros2, firmware, bus]
status: complete
updated: 2026-07-21
summary: "硬件通信与协议专题汇总：从电机驱动固件、现场总线（EtherCAT/CAN/UART）到 ROS 2 / LCM 软件中间件，覆盖人形与移动机器人底层数据链路选型。"
---

# 硬件通信与协议（专题汇总）

> **图谱专题视图**：本页是知识图谱「🔌 通信协议 (Communication)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=communication) 筛选时，本节点为汇总锚点。

## 一句话定义

**通信协议专题** 回答机器人 **关节驱动、传感器与上层控制器之间** 用什么物理层/协议传数据，以及如何在延迟、带宽、同步与生态之间选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网实时现场总线 |
| CAN | Controller Area Network | 车载/关节常用串行总线 |
| CAN-FD | CAN with Flexible Data-Rate | 更高带宽 CAN 变体 |
| ROS 2 | Robot Operating System 2 | 机器人软件中间件（DDS 传输） |
| LCM | Lightweight Communications Marshaling | 轻量 pub/sub，常用于低延迟控制 |

## 为什么重要

- **控制环路吃延迟**：1 kHz 力控下，总线抖动会直接表现为抖动/啸叫。
- **软硬件分层**：同一策略可在 ROS 2 跑规划，在 EtherCAT 跑关节伺服。
- **V21 硬件链路形式化**：本库把「驱动固件 → 总线 → 中间件」作为独立知识链维护。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 总览 | 驱动-固件-总线栈 | [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md) |
| 现场总线 | EtherCAT / CAN 选型 | [EtherCAT Protocol](../concepts/ethercat-protocol.md)、[CAN vs EtherCAT](../comparisons/can-vs-ethercat-joint-bus.md) |
| 串口层 | RS-485 / UART | [RS-485](../concepts/rs-485-serial-bus.md)、[UART](../concepts/uart-serial-communication.md) |
| 中间件 | ROS 2 vs LCM | [ROS2 Basics](../concepts/ros2-basics.md)、[ROS2 vs LCM](../comparisons/ros2-vs-lcm.md) |
| DDS | ROS 2 底层 QoS/RTPS | [DDS 通信机制](../concepts/dds-communication.md) |
| 时钟 | 分布式同步 | [Clock Synchronization](../concepts/clock-synchronization-algorithms.md) |
| 系统工程 | OS/边云/OTA/安全 FSM | [系统工程专题](./topic-systems-engineering.md) |

## 与其他专题的关系

- **[触觉](./topic-tactile.md)**：力控环对总线延迟敏感。
- **[WBC](./topic-wbc.md)**：全身控制在实时层需稳定关节接口。
- **[状态估计](./topic-state-estimation.md)**：多传感器时间对齐依赖时钟同步。

## 关联页面

- [Field-Oriented Control](../concepts/field-oriented-control.md)
- [EtherCAT vs EtherNet/IP](../comparisons/ethercat-vs-ethernet-ip.md)
- [ROS2 vs LCM](../comparisons/ros2-vs-lcm.md)

## 参考来源

- 本库归纳自 [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md) 及 `wiki/concepts/*protocol*` 系列页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`communication` 命中规则）
