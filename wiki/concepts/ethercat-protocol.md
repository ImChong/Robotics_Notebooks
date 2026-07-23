---
type: concept
tags: [middleware, hardware, protocol, ethercat, realtime]
status: complete
updated: 2026-04-21
related:
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../queries/ethercat-master-optimization.md
  - ../comparisons/ethercat-vs-ethernet-ip.md
  - ../comparisons/can-vs-ethercat-joint-bus.md
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ./can-bus-protocol.md
  - ./can-fd.md
  - ../queries/real-time-control-middleware-guide.md
  - ../formalizations/control-loop-latency-modeling.md
  - ./clock-synchronization-algorithms.md
sources:
  - ../../sources/papers/sim2real.md
  - ../../sources/sites/cia_canopen_overview.md
summary: "EtherCAT 协议基础：介绍了这款基于以太网的高性能工业现场总线协议，重点讲解了其“运行中处理”机制与分布式时钟（DC）如何支撑起人形机器人的硬实时控制。"
---

# EtherCAT 协议基础

**EtherCAT (Ethernet for Control Automation Technology)** 是目前人形机器人底层总线的首选协议。它解决了标准以太网因冲突检测（CSMA/CD）而无法保证实时性的问题，将带宽利用率发挥到了物理极限。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |

## 核心机制：On-the-fly 处理

不同于普通网络每个节点都要接收并解析报文，在 EtherCAT 中：
1. 主站发出的以太网帧像一列火车一样穿过所有从站。
2. 每个从站（电机驱动器）在报文经过的瞬间，**“在运行中”** 提取属于自己的指令，并把自己当前的位置/扭矩数据塞入报文对应的位置。
3. 报文到达链尾后原路折返。

这使得 100 个轴的刷新周期可以轻松突破 **250微秒**。

## 关键特性

- **分布式时钟 (DC)**：支持多轴之间纳秒级的同步精度。
- **拓扑灵活**：支持线型、树型、星型拓扑，对人形机器人布线极其友好。
- **协议栈**：
  - **CoE (CANopen over EtherCAT)**：最主流的应用层协议。

## 关联页面
- [执行器驱动链选型闭环知识链](../queries/actuator-drive-chain-selection-loop.md) — 本页处于驱动链 **④层 实时总线闭环集成**（总线周期 ≠ 闭环带宽、抖动确定性）
- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md)
- [CAN vs EtherCAT：关节总线选型](../comparisons/can-vs-ethercat-joint-bus.md)
- [CAN 总线（经典）](./can-bus-protocol.md)
- [EtherCAT 主站优化指南](../queries/ethercat-master-optimization.md)
- [EtherCAT vs EtherNet/IP（工业总线选型对比）](../comparisons/ethercat-vs-ethernet-ip.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)
- [时钟同步算法](./clock-synchronization-algorithms.md)

## 参考来源
- EtherCAT Technology Group (ETG) 官方白皮书。
