# RS-485（TIA/EIA-485）一手资料索引

> 来源归档（ingest）

- **标题：** RS-485 / TIA-485 差分多点总线与接口设计应用指南
- **类型：** standard / application note（合集）
- **入库日期：** 2026-06-10
- **一句话说明：** 汇总 TIA-485-A 电气标准、TSB-89 应用指南与 TI SLLA383/SLLA070 收发器设计笔记，作为 `wiki/concepts/rs-485-serial-bus.md` 的原始依据。
- **沉淀到 wiki：** 是 → [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)、[uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

## 为什么值得保留

- RS-485 是机器人 **底盘、温湿度、老款伺服、Modbus RTU 外设** 的常见物理层；与 CAN 不同，**无硬件仲裁**，半双工方向控制与终端偏置是现场可靠性的关键。
- 一手资料覆盖：**单位负载、差分阈值、A/B 极性「Pesky Polarity」、总线拓扑与波特率–距离积**。

## 核心摘录

### 1) TIA/EIA-485-A — 正式标准

- **来源：** ANSI/TIA/EIA-485-A-1998 (Reaffirmed 2012)，*Electrical Characteristics of Generators and Receivers for Use in Balanced Digital Multipoint Systems*；概述见 [Wikipedia: RS-485](https://en.wikipedia.org/wiki/RS-485)。
- **要点：**
  - 仅定义 **物理层**（发生器/接收器电气特性），**不定义** 通信协议；Modbus RTU、Profibus DP 等运行在 RS-485 之上。
  - **差分对 A、B**（及可选信号地 C）：驱动器在 54 Ω 负载上差分 ≥ 1.5 V；接收器检测阈值 ±200 mV。
  - **三态驱动**：允许多个发送器挂总线，同一时刻仅一个驱动；典型 **≥ 32 单位负载** 节点。
  - 逻辑状态：标准以 A 相对 B 的极性定义 mark/space，**不分配** UART 逻辑功能——与芯片厂商 A/B 标注可能需交叉核对。
  - 共模范围：接收端允许约 **−7 V 至 +12 V** 共模（相对本地地）；长距离地电位差需限制 SC/GND 环路电流。
- **对 wiki 的映射：** [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)

### 2) TIA TSB-89A — 应用指南

- **来源：** [TSB-89A — Application Guidelines for TIA/EIA-485-A](https://www.tiaonline.org/)（应用指南 PDF，常被标准前言引用）
- **要点：**
  - **线性总线（multidrop）** 优于星形/环形；支路（stub）过长会引起反射。
  - **经验法则**：比特率 × 线缆长度（m）≤ 10<sup>8</sup>（如 50 m 电缆建议 ≤ 2 Mbit/s）；更高速度需更短距离。
  - **终端电阻**：两端各一只，阻值等于特性阻抗（双绞线常 **120 Ω**）。
  - **偏置（bias）**：空闲总线需拉到确定差分态，避免噪声误判；具体阻值由器件厂商 AN 给出（标准不强制单一数值）。
- **对 wiki 的映射：** [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)

### 3) Texas Instruments SLLA383 — UART 转 RS-485 接口

- **来源：** [SLLA383B — UART-to-RS-485 Interface](https://www.ti.com/lit/an/slla383b/slla383b.pdf)
- **要点：**
  - **DE（Driver Enable）/ RE̅（Receiver Enable）** 控制半双工方向：发送前使能驱动、关闭接收；发送后反转。
  - MCU UART **全双工引脚** 经收发器映射到 **半双工总线**；固件或 RTS 引脚需编排 turnaround 时间。
  - 自动方向控制收发器可减轻 GPIO 负担，但仍有 **位间切换延迟** 约束。
- **对 wiki 的映射：** [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)

### 4) Texas Instruments SLLA070 — RS-485 标准概览与配置

- **来源：** [SLLA070D — RS-422 and RS-485 Standards Overview and System Configurations](https://www.ti.com/lit/an/slla070d/slla070d.pdf)
- **要点：**
  - RS-485 与 RS-422 均用差分，但 RS-422 常为 **单驱动多点接收**；RS-485 为 **多点半双工**。
  - **全双工四线** RS-485 可行（两对差分），但机器人现场更常见半双工两线 + 协议主从轮询。
  - **A/B 极性**：多家收发器数据手册与 Profibus 布线颜色约定不一致，布线时应 **以差分电压极性为准** 而非仅看丝印。
- **对 wiki 的映射：** [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)、[rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)

### 5) Modbus — RS-485 上最常见应用层之一

- **来源：** Modbus Organization [Modbus over serial line specification](https://modbus.org/docs/Modbus_over_serial_line_V1_02.pdf)
- **要点：**
  - **RTU** 模式：8/N/1 或 8/E/1 常见；主站轮询、从站地址 1–247；CRC16 帧尾。
  - 物理层 RS-485 + 应用层 Modbus 是 **底盘电机、PLC I/O** 的经典组合，与关节 CAN 私有协议分工不同。
- **对 wiki 的映射：** [rs-485-serial-bus](../../wiki/concepts/rs-485-serial-bus.md)、[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

## 推荐继续阅读（外部）

- TI [SLLA847 — How Far and How Fast Can You Go with RS-485?](https://www.ti.com/lit/an/slla847/slla847.pdf)
- Analog Devices [RS-485 Circuit Implementation Guide](https://www.analog.com/en/resources/analog-dialogue/articles/rs-485-circuit-implementation-guide.html)

## 当前提炼状态

- [x] 摘要与 wiki 映射
