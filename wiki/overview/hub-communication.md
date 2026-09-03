---
type: overview
tags: [hub, hub-communication, ethercat, can, ros2, firmware, bus, rpc, grpc]
status: complete
updated: 2026-09-03
summary: "硬件通信与协议知识链汇总：从电机驱动固件、现场总线（EtherCAT/CAN/UART）到 ROS 2 / LCM / RPC 软件中间件，覆盖人形与移动机器人底层数据链路选型。"
---

# 硬件通信与协议（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=humanoid-hardware-design) 与 [路线页](../../roadmap/depth-humanoid-hardware-design.md)。

## 一句话定义

**通信协议知识链** 回答机器人 **关节驱动、传感器与上层控制器之间** 用什么物理层/协议传数据，以及如何在延迟、带宽、同步与生态之间选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网实时现场总线 |
| CAN | Controller Area Network | 车载/关节常用串行总线 |
| CAN-FD | CAN with Flexible Data-Rate | 更高带宽 CAN 变体 |
| ROS 2 | Robot Operating System 2 | 机器人软件中间件（DDS 传输） |
| LCM | Lightweight Communications Marshaling | 轻量 pub/sub，常用于低延迟控制 |
| RPC | Remote Procedure Call | 请求/响应远程调用；见 [RPC 概念](../concepts/remote-procedure-call.md) |
| gRPC | gRPC Remote Procedure Calls | 现代 RPC 框架（HTTP/2）；见 [gRPC](../entities/grpc.md) |

## 为什么重要

- **控制环路吃延迟**：1 kHz 力控下，总线抖动会直接表现为抖动/啸叫。
- **软硬件分层**：同一策略可在 ROS 2 跑规划，在 EtherCAT 跑关节伺服。
- **V21 硬件链路形式化**：本库把「驱动固件 → 总线 → 中间件」作为独立知识链维护。

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 总览 | 驱动-固件-总线栈 | [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md) |
| 现场总线 | EtherCAT / CAN 选型 | [EtherCAT Protocol](../concepts/ethercat-protocol.md)、[CAN vs EtherCAT](../comparisons/can-vs-ethercat-joint-bus.md) |
| 串口层 | RS-485 / UART | [RS-485](../concepts/rs-485-serial-bus.md)、[UART](../concepts/uart-serial-communication.md) |
| 进程间通信 | POSIX 原语与分层选型 | [IPC 基础](../concepts/ipc-inter-process-communication.md) |
| 中间件 | ROS 2 vs LCM | [ROS2 Basics](../concepts/ros2-basics.md)、[ROS2 vs LCM](../comparisons/ros2-vs-lcm.md) |
| DDS | ROS 2 底层 QoS/RTPS | [DDS 通信机制](../concepts/dds-communication.md) |
| RPC | 请求/响应服务面 | [远程过程调用](../concepts/remote-procedure-call.md)、[gRPC](../entities/grpc.md) |
| 时钟 | 分布式同步 | [Clock Synchronization](../concepts/clock-synchronization-algorithms.md) |
| 整机架构 | 分域/拓扑/延迟预算 | [机器人整机通信架构](../concepts/robot-onboard-communication-architecture.md) |
| 系统工程 | OS/边云/OTA/安全 FSM | [系统工程知识链](./hub-systems-engineering.md) |

## 与其他知识链的关系

- **[触觉](./hub-tactile.md)**：力控环对总线延迟敏感。
- **[WBC](./hub-wbc.md)**：全身控制在实时层需稳定关节接口。
- **[状态估计](./hub-state-estimation.md)**：多传感器时间对齐依赖时钟同步。

## 关联页面

- [进程间通信（IPC）](../concepts/ipc-inter-process-communication.md)
- [机器人整机通信架构](../concepts/robot-onboard-communication-architecture.md) · [人形整机硬件设计纵深路线](../../roadmap/depth-humanoid-hardware-design.md)
- [Field-Oriented Control](../concepts/field-oriented-control.md)
- [EtherCAT vs EtherNet/IP](../comparisons/ethercat-vs-ethernet-ip.md)
- [ROS2 vs LCM](../comparisons/ros2-vs-lcm.md)

## 参考来源

- 本库归纳自 [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md) 及 `wiki/concepts/*protocol*` 系列页
- RPC 一手：[Birrell & Nelson 1984](../../sources/papers/birrell_nelson_implementing_rpc_tocs_1984.md) · [RFC 5531](../../sources/sites/rfc-5531-onc-rpc.md) · [gRPC 文档](../../sources/sites/grpc-io-docs.md)
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`communication` 命中规则）
