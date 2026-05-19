---
type: comparison
tags: [hardware, fieldbus, can-bus, can-fd, ethercat, uart, rs485, realtime, robotics]
status: complete
updated: 2026-05-19
related:
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ../concepts/can-bus-protocol.md
  - ../concepts/can-fd.md
  - ../concepts/ethercat-protocol.md
  - ../concepts/uart-serial-communication.md
  - ../comparisons/ethercat-vs-ethernet-ip.md
  - ../formalizations/control-loop-latency-modeling.md
sources:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_can_fd_basic_idea.md
  - ../../sources/courses/uart_rs485_serial_embedded.md
summary: "腿式与人形关节反馈选型：经典 CAN/CAN FD、EtherCAT 与 UART/RS485 在带宽、拓扑、成本、生态与硬实时上的对照，帮助判断何时从 CAN 升级到 EtherCAT 或保留串口作调试与外设。"
---

# CAN / CAN FD vs EtherCAT vs UART·RS485（关节与现场总线选型）

> 对比轴：**多轴硬实时关节环**、**中等轴数成本敏感平台**、**调试与外设** 三类需求。

## 一句话结论

- **≥20 轴、250 µs–1 ms 级同步**：优先 **EtherCAT**（或同等工业以太网现场总线）。
- **中等轴数、已有 CAN 电机生态、成本敏感**：**经典 CAN** 或 **CAN FD** + 合理分包仍够用。
- **日志、遥控、单传感器、产测**：**UART / RS485**；不要承担全关节 1 kHz 闭环。

## 核心对比表

| 维度 | 经典 CAN | CAN FD | EtherCAT | UART / RS485 |
|------|----------|--------|----------|----------------|
| 物理介质 | CAN_H/L 差分 | 同左，FD 收发器 | 100BASE-TX 以太网 | TTL / ±RS-232 / RS-485 差分 |
| 典型速率 | ≤ 1 Mbit/s | 数据段 2–8 Mbit/s 级 | 100 Mbit/s 级管道 | 9600 – 3 M+ baud |
| 单帧载荷 | ≤ 8 B | ≤ 64 B | 以太网帧内多子报文 | 流式字节 |
| 多节点仲裁 | 硬件按 ID | 同左 | 主站调度 + on-the-fly | 无（RS485 需主从协议） |
| 多轴时间同步 | 软件时间戳 | 同左 | **DC 纳秒级** | 通常无 |
| 100 轴刷新 | 紧张，需规划 | 明显改善 | 可达 **250 µs** 级 | 不适用 |
| 线缆/拓扑 | 线型 + 120 Ω | 同左，更严 | 线/树/星 | RS485 总线 + 终端 |
| 机器人常见度 | 四足/人形电机 | 新驱动器 | 人形高端、工业臂 | 调试、IMU、遥控 |
| 应用层 | CANopen、DroneCAN、私有 | CANopen FD | CoE 等 | Modbus RTU、自定义 |

## 何时从 CAN 升级到 EtherCAT

- 控制频率 × 轴数 × 每轴反馈字节数 **逼近总线利用率 50–70%** 且延迟长尾可见。
- 需要 **分布式时钟** 做相位对齐力矩（步行、全身 WBC）。
- 已有 EtherCAT 驱动器供应链，主站栈（SOEM、IgH 等）可接受。

## 何时 CAN FD 而非换 EtherCAT

- 现有 **CAN 布线/驱动器** 可固件升级 FD，希望 **减少换栈成本**。
- 轴数中等（例如 ≤12–20），FD 组帧后总线裕量足够。

## UART/RS485 的正确定位

- **不要**与 CAN/EtherCAT 争关节闭环带宽。
- **要**用于：USB 串口调试、无线遥控接收、单点 Modbus 传感器、产测命令。

## 常见误判

| 误判 | 实际情况 |
|------|----------|
| USB2CAN 等于板载 CAN 实时性 | USB 主机调度引入抖动，宜仅用于标定与非实时命令 |
| CAN FD 自动解决所有轴数问题 | 仲裁段仍 1 Mbit/s 级，极高轴数仍可能撞 EtherCAT |
| RS485 拉长即可替代 CAN | 无硬件仲裁与标准化错误处理，协议设计负担重 |

## 关联页面

- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md)
- [CAN 总线](./../concepts/can-bus-protocol.md)
- [CAN FD](./../concepts/can-fd.md)
- [EtherCAT 协议基础](./../concepts/ethercat-protocol.md)
- [UART 串行通信](./../concepts/uart-serial-communication.md)
- [EtherCAT vs EtherNet/IP](./ethercat-vs-ethernet-ip.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)

## 参考来源

- [CiA：经典 CAN 与物理层](../../sources/sites/cia_can_knowledge_can_classic_and_hs.md)
- [CiA：CAN FD](../../sources/sites/cia_can_fd_basic_idea.md)
- [UART / RS-485 入门](../../sources/courses/uart_rs485_serial_embedded.md)

## 推荐继续阅读

- [EtherCAT vs EtherNet/IP](./ethercat-vs-ethernet-ip.md)
- CiA [Designing a CAN network](https://www.can-cia.org/can-knowledge/designing-a-can-network/)
