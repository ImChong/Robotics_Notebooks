---
type: concept
tags: [middleware, hardware, protocol, can-bus, fieldbus, embedded, robotics]
status: complete
updated: 2026-05-19
related:
  - ./can-fd.md
  - ./ethercat-protocol.md
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ../comparisons/can-vs-ethercat-joint-bus.md
  - ../formalizations/control-loop-latency-modeling.md
  - ../concepts/processor-in-the-loop-sim2real.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_canopen_overview.md
  - ../../sources/sites/cia_dronecan_uavcan.md
summary: "经典 CAN（Controller Area Network）总线：多主、按位仲裁的差分串行现场总线，1 Mbit/s 与 8 字节载荷上限，广泛用于腿式/人形关节驱动与车载 ECU，应用层常见 CANopen 或 DroneCAN。"
---

# CAN 总线（经典 CAN / CAN 2.0）

**CAN（Controller Area Network）** 是一种 **多主、广播式、带硬件仲裁** 的串行现场总线。在机器人里，它最常见于 **主控板 ↔ 关节电机驱动器** 的反馈与力矩指令链路；在汽车里则是 ECU 组网的事实标准。

## 一句话定义

多条节点共享一对差分线（CAN_H / CAN_L），任意节点可发起帧；若 ID 冲突，**数值越小（越 dominant）的标识符赢得总线**——无需中央主站即可安全共享介质。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| SDK | Software Development Kit | 软件开发工具包 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线 |

## 为什么重要

- **成本与生态**：收发器与 MCU 内置 CAN 控制器便宜，开源人形/四足 BOM 里常见 **USB2CAN**、板载 CAN 走线。
- **硬实时友好（相对以太网）**：确定性仲裁、短帧、无 TCP 握手；但仍受 **1 Mbit/s、8 byte 载荷** 与线长约束，高轴数时需规划 ID 与分包策略。
- **Sim2Real 隐形杀手**：真机失败常来自 **CAN 迟到、解析错误、周期错位**，而非仅摩擦系数——见 [处理器在环 Sim2Real](./processor-in-the-loop-sim2real.md)。

## 核心机制

### 1. 帧与仲裁

- 标准帧 11 bit ID / 扩展帧 29 bit ID；数据场 **0–8 byte**。
- **按位仲裁**：发送节点同时监听总线，发隐性位却读到显性位则退避——低 ID 优先。
- **错误帧与 ACK**：硬件重发机制减轻瞬时干扰，但 **总线利用率过高** 仍会导致延迟长尾。

### 2. 物理层（CAN HS）

- ISO 11898-2 **高速收发器**：隐性 ~2.5 V，显性差分约 0.9–2.0 V。
- 位速率 **≤ 1 Mbit/s**；速率越高，允许网线越短（线型拓扑 + **两端 120 Ω** 终端为常规做法）。
- 线缆传播延迟建议 ≤ 5 ns/m（CiA 推荐量级）。

### 3. 应用层：你实际在用的往往不止 CAN

| 协议 | 典型场景 |
|------|----------|
| **CANopen**（CiA 301/402） | 工业伺服、关节驱动器对象字典、PDO/SDO |
| **DroneCAN** | ArduPilot / PX4 与 CAN 外设（ESC、传感器） |
| **厂商私有协议** | 部分人形/四足电机 SDK 的紧凑二进制帧 |
| **CoE** | 在 EtherCAT 上传 CANopen 对象（见 [EtherCAT](./ethercat-protocol.md)） |

## 常见误区

- **「CAN 就是 1 Mbps 随便接」**：拓扑、终端、接地与采样点不一致都会让「理论上 1 M」在真机上变成间歇 NACK。
- **「用 USB 转 CAN 就等同板载实时」**：USB 调度与批量传输会引入 **毫秒级抖动**，不适合单独承担 500 Hz–1 kHz 全关节闭环（宜板载 CAN + RT 线程）。
- **混淆 CAN 与 UART/RS485**：UART 无硬件仲裁；RS485 半双工需方向控制，协议栈完全不同——见 [UART 串行通信](./uart-serial-communication.md)。

## 关联页面

- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md) — CANopen / 私有帧 / CiA 402 等应用层选型
- [CAN FD（Flexible Data Rate）](./can-fd.md)
- [EtherCAT 协议基础](./ethercat-protocol.md)
- [CAN vs EtherCAT：关节总线选型](../comparisons/can-vs-ethercat-joint-bus.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [CiA：经典 CAN、HS 物理层与历史](../../sources/sites/cia_can_knowledge_can_classic_and_hs.md)
- [CiA：CANopen 概览](../../sources/sites/cia_canopen_overview.md)
- [CiA / DroneCAN：无人机 CAN 应用层](../../sources/sites/cia_dronecan_uavcan.md)

## 推荐继续阅读

- CiA [CAN knowledge](https://www.can-cia.org/can-knowledge/) 目录
- ISO 11898 系列（经典 CAN 数据链路与物理层）
