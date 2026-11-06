---
type: overview
tags: [hub, hub-communication, ethercat, can, ros2, firmware, bus, rpc, grpc, mqtt, iot, zeromq]
status: complete
original:
  - ../../sources/papers/ethercat_reading_arxiv_2401_08583_ec1.md
  - ../../sources/papers/lcm_reading_iros2010_rtcontrol_middleware_reading_a7.md
  - ../../sources/sites/cia_can_canopen_canfd_hs_reading_mdfbp1.md
updated: 2026-09-17
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
| MQTT | Message Queuing Telemetry Transport | IoT pub/sub；Broker 路由；见 [MQTT 协议](../concepts/mqtt-protocol.md) |
| ZMQ | ZeroMQ | Brokerless socket 消息库；见 [ZeroMQ](../entities/zeromq.md) |

## 为什么重要

- **控制环路吃延迟**：1 kHz 力控下，总线抖动会直接表现为抖动/啸叫。
- **软硬件分层**：同一策略可在 ROS 2 跑规划，在 EtherCAT 跑关节伺服。
- **V21 硬件链路形式化**：本库把「驱动固件 → 总线 → 中间件」作为独立知识链维护。

## 核心原理

按「物理层/总线 → 应用协议 → 软件中间件」三层理解（依据三份已通读归档）：

- **CAN / CAN FD**：CANopen 基于 ISO 11898-1 数据链路层，设备由协议栈、应用软件、对象字典三部分组成；CAN FD 把载荷由 8 字节扩至 64 字节，并用 BRS 位在数据段切换到第二速率（来源：CiA 官网三页）。CAN HS 物理层最高 1 Mbit/s，线越长速率越低，推荐线型拓扑与两端 120 Ω 终端。
- **EtherCAT**：标准以太网卡 + 主站软件，典型周期 <100 µs、抖动 <1 µs（论文转述 ETG）；脊柱手术机器人案例用 IgH 主站 + PREEMPT_RT 在 1 kHz 下实测 3 小时，周期 1000.017 ± 7.035 µs、最大抖动 148 µs。
- **LCM（对比 ROS）**：UDP 组播、无中心 hub，尽力投递、宁可丢包也不延迟后续消息，每条消息带 64 位类型指纹防止类型不一致（来源：LCM IROS 2010）。

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
| IoT / 遥测 | 边云状态、远程运维 | [MQTT 协议](../concepts/mqtt-protocol.md)、[Mosquitto](../entities/mosquitto.md) |
| 策略 Server / 调试流 | GPU 推理进程 ↔ 仿真/真机；时序工具 | [ZeroMQ 消息模式](../concepts/zeromq-messaging.md)、[ZeroMQ](../entities/zeromq.md)、[PlotJuggler](../entities/plotjuggler.md) |
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

## 工程实践

- 关节环（1 kHz 量级）优先选确定性总线（EtherCAT 或 CAN/CAN-FD），上层规划/感知再走 ROS 2 / LCM（此为本库归纳，推断）。
- 总线选型先看周期、抖动、节点数与布线长度：CAN 受线长-速率制约，EtherCAT 需带标准网卡的实时主站（PREEMPT_RT 单内核方案可行，见上述案例）。
- 遥测/调试流用 MQTT、ZeroMQ 等；请求/响应服务面用 RPC/gRPC，详见上文覆盖表。

## 局限

- 上述实测数据来自单一小型案例（3 台从站、1 kHz），LCM 实验在 2009–2010 年通用 Linux 上完成，均未覆盖人形全身数十关节的规模，结论需在目标硬件上复测（推断）。
- 本页对 DDS/ROS 2 QoS、gRPC、MQTT、ZeroMQ 的论述由对应子页与参考来源支撑，不由三份通读归档覆盖；CiA 301/402 等规范本体未读。

## 继续阅读

- [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md)
- [EtherCAT Protocol](../concepts/ethercat-protocol.md) · [CAN vs EtherCAT](../comparisons/can-vs-ethercat-joint-bus.md) · [ROS2 vs LCM](../comparisons/ros2-vs-lcm.md)

## 参考来源

- 本库归纳自 [Motor Drive / Firmware / Bus Protocols](./motor-drive-firmware-bus-protocols.md) 及 `wiki/concepts/*protocol*` 系列页
- RPC 一手：[Birrell & Nelson 1984](../../sources/papers/birrell_nelson_implementing_rpc_tocs_1984.md) · [RFC 5531](../../sources/sites/rfc-5531-onc-rpc.md) · [gRPC 文档](../../sources/sites/grpc-io-docs.md)
- ZeroMQ 一手：[zeromq.org](../../sources/sites/zeromq-org-primary-refs.md) · [ØMQ Guide](../../sources/sites/zguide-zeromq.md) · [ZMTP RFC](../../sources/sites/zmq-rfc-zmtp-3.md) · [libzmq](../../sources/repos/libzmq.md)
- 通读归档：[EtherCAT 论文](../../sources/papers/ethercat_reading_arxiv_2401_08583_ec1.md) · [LCM 论文](../../sources/papers/lcm_reading_iros2010_rtcontrol_middleware_reading_a7.md) · [CiA CAN 三页](../../sources/sites/cia_can_canopen_canfd_hs_reading_mdfbp1.md)
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`communication` 命中规则）
