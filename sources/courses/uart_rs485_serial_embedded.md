# UART / RS-232 / RS-485 异步串行通信（嵌入式与机器人）

> 来源归档（course / 技术入门合集）

- **标题：** UART 与工业串行总线入门资料索引
- **类型：** course
- **来源：**
  - Wikipedia：[Universal asynchronous receiver-transmitter](https://en.wikipedia.org/wiki/Universal_asynchronous_receiver-transmitter)
  - Texas Instruments 应用笔记：[SLLA383 — UART-to-RS-485 Interface](https://www.ti.com/lit/an/slla383b/slla383b.pdf)（PDF，200 可访问）
  - Microsoft Learn（架构参考）：[UART architecture](https://learn.microsoft.com/en-us/windows-hardware/drivers/serports/uart-architecture)
- **入库日期：** 2026-05-19
- **一句话说明：** 板间点对点调试、IMU/遥控器/旧款驱动器常用的异步串行栈：UART 帧格式 + 电平标准（TTL/RS-232/RS-485）与机器人布线注意点。

## 为什么值得保留

- **与 CAN 互补**：CAN 适合多节点仲裁总线；UART/RS485 适合 **少量节点、低成本、长距离半双工**（如底盘、云台、外设模组）。
- **调试入口**：几乎所有 MCU 出厂带 UART，是 bring-up、日志与 Bootloader 的第一接口。
- **Sim2Real 侧链**：IMU、遥控接收机等常走 UART/SPI/I2C，与 [processor-in-the-loop-sim2real](../../wiki/concepts/processor-in-the-loop-sim2real.md) 中的 I2C 路径并列。

## 核心摘录

### 1) UART 是什么

- **要点：** **UART（Universal Asynchronous Receiver-Transmitter）** 为可配置波特率与数据格式的 **异步** 串行外设；按位发送 LSB→MSB，由起始/停止位界定帧，时序由通信双方约定。电平由外部驱动实现：常见 **TTL**、**RS-232**、**RS-485**。
- **对 wiki 的映射：** [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

### 2) USART 与同步模式

- **要点：** **USART** 在 UART 基础上增加同步时钟线能力；机器人固件中「串口」多数指异步 UART。
- **对 wiki 的映射：** [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

### 3) RS-485 与多点总线

- **要点：** RS-485 为 **差分、半双工** 物理层，支持一条总线上多驱动器/接收器（典型 ≤32 单元），常用于 Modbus RTU、部分伺服与传感器组网；需 **终端匹配与偏置电阻** 避免空闲态误码。TI SLLA383 等资料给出 UART↔RS485 收发器接口与方向控制（DE/RE）要点。
- **对 wiki 的映射：** [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)、[can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)（选型语境）

### 4) 与 CAN 的分工（工程直觉）

- **要点：** UART 无硬件仲裁，多主机需软件协议；CAN 内置仲裁与错误帧。高轴数、硬实时关节反馈优先 CAN/CAN FD/EtherCAT；低速传感、人机接口、产测串口优先 UART。
- **对 wiki 的映射：** [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)

## 推荐继续阅读（外部）

- TI SLLA383B PDF（RS-485 接口设计）
- Modbus Organization：RTU over serial line

## 当前提炼状态

- [x] 摘要与 wiki 映射
