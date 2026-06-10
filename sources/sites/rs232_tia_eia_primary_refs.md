# RS-232（TIA/EIA-232）一手资料索引

> 来源归档（ingest）

- **标题：** RS-232 / TIA-232 串行接口标准与 ITU-T 对应规范
- **类型：** standard / application note（合集）
- **入库日期：** 2026-06-10
- **一句话说明：** 汇总 TIA-232-F 正式标准、ITU-T V.24/V.28 信号定义与 Maxim/TI 接口设计指南，作为 `wiki/concepts/rs-232-serial-interface.md` 的原始依据。
- **沉淀到 wiki：** 是 → [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)、[uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

## 为什么值得保留

- RS-232 仍是 **工控 CNC、PLC、老式仪器、产测治具** 与 PC 通信的常见物理层；与 TTL 电平、RS-485 差分总线在电压、距离、连接器上差异显著，需以 TIA 标准为准。
- 一手资料强调：**数据电路 mark = 负电压、space = 正电压**，与 TTL 正逻辑相反——这是接线与电平转换芯片选型的常见坑。

## 核心摘录

### 1) TIA-232-F — 正式标准（现行）

- **来源：** ANSI/TIA/EIA-232-F (R2012)，*Interface Between Data Terminal Equipment and Data Circuit-Terminating Equipment Employing Serial Binary Data Interchange*；TIA [TR-30 委员会](https://www.tiaonline.org/) 维护。概述见 [Wikipedia: RS-232](https://en.wikipedia.org/wiki/RS-232)。
- **要点：**
  - 定义 **DTE（数据终端设备）** 与 **DCE（数据电路终端设备）** 之间的电气特性、时序、连接器与信号功能。
  - **数据电路**（TxD/RxD）：逻辑 1（mark）= −3 V 至 −15 V；逻辑 0（space）= +3 V 至 +15 V（相对信号地）。
  - **控制电路**（RTS/CTS/DTR/DSR 等）极性与数据电路相反：asserted = 正电压。
  - 标准 **不规定** 字符编码、帧格式、波特率上限（仅说明适用于 &lt; 20 kbit/s 量级）；位格式由 UART 硬件配置。
  - 电缆电容限制驱动能力；经验法则常用 **≤ 15 m @ 标准电缆**，低电容线可更长。
- **对 wiki 的映射：** [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)

### 2) ITU-T V.24 / V.28 — 国际对应与信号定义

- **来源：** ITU-T [V.24](https://www.itu.int/rec/T-REC-V.24)（电路定义）、[V.28](https://www.itu.int/rec/T-REC-V.28)（非平衡电气特性）；TIA-232-E 起与 V.24/V.28 对齐。
- **要点：**
  - V.24 列出 **电路编号与功能**（如 103=TxD、104=RxD、105=RTS、106=CTS、107=RTS 等变体）。
  - V.28 规定 **单端、非平衡** 接口的电压与阻抗；与 RS-232 驱动/接收芯片（如 MAX232、SP3232）数据手册一致。
- **对 wiki 的映射：** [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)

### 3) Maxim Integrated — RS-232 接口设计指南

- **来源：** Maxim [RS-232 and RS-485 Design Guide](https://www.maximintegrated.com/en/design/technical-documents/tutorials/1/2146.html)（原 Maxim AN2146 系列）
- **要点：**
  - **电荷泵** RS-232 收发器（±5 V 或 ±10 V 摆幅）可由 3.3 V/5 V 单电源供电，是 MCU TTL 与 DB9 之间的标准桥梁。
  - **3 线制**（TxD/RxD/GND）在嵌入式调试中足够；全信号 25/9 针用于调制解调器时代的手握协议。
  - **Null modem** 交叉 TxD/RxD 使两台 DTE 直连，属工程惯例而非标准正文。
- **对 wiki 的映射：** [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)

### 4) EIA 历史修订与连接器

- **来源：** Wikipedia RS-232 历史节；TIA-232-D 起 **DE-9** 成为 PC 常见连接器；DB-25 为早期完整信号集。
- **要点：**
  - DE-9 **DTE 公头** 引脚：2=RxD、3=TxD、5=GND 为最小集；5=CTS、7=RTS 用于硬件流控。
  - 工业现场存在 **非标准电压**（±5 V「兼容 RS-232」）与 **RS-232 电平标签的 TTL 设备** 混用，需用示波器或数据手册核实。
- **对 wiki 的映射：** [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)、[ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)

## 推荐继续阅读（外部）

- TI [SLLA038 — RS-232 Basics](https://www.ti.com/lit/an/slla038/slla038.pdf)
- SparkFun [Serial Communication](https://learn.sparkfun.com/tutorials/serial-communication) — RS-232 与 TTL 对比

## 当前提炼状态

- [x] 摘要与 wiki 映射
