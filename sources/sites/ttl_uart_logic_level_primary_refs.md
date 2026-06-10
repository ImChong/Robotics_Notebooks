# TTL / CMOS UART 逻辑电平（一手资料索引）

> 来源归档（ingest）

- **标题：** TTL 与 CMOS 逻辑电平、UART 板内串行接口一手资料
- **类型：** standard / datasheet / textbook（合集）
- **入库日期：** 2026-06-10
- **一句话说明：** 汇总 JEDEC 逻辑电平规范、经典逻辑家族数据手册与 UART 架构参考，作为 `wiki/concepts/ttl-serial-logic-level.md` 的原始依据。
- **沉淀到 wiki：** 是 → [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)、[uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

## 为什么值得保留

- 机器人固件里说的「串口」在板级几乎总是 **MCU UART 引脚上的 TTL/CMOS 单端电平**，而非 RS-232 的 ± 电平；一手资料可避免把「逻辑 1/0」与 RS-232 的 mark/space 极性混为一谈。
- TTL 并非像 TIA-485 那样的单一通信标准，而是 **逻辑家族电压规范 + UART 帧格式** 的组合；需引用 JEDEC 与器件数据手册而非二手博客。

## 核心摘录

### 1) JEDEC — 逻辑电平与输入/输出兼容性

- **来源：** JEDEC Solid State Technology Association，逻辑电平与接口标准系列（如 JESD8 等）；行业惯例引用见 [Wikipedia: Logic level](https://en.wikipedia.org/wiki/Logic_level)、[Wikipedia: Transistor–transistor logic](https://en.wikipedia.org/wiki/Transistor%E2%80%93transistor_logic)。
- **要点：**
  - **经典 5 V TTL**：逻辑 1 输出典型 ≥ 2.4 V，逻辑 0 ≤ 0.4 V；输入高电平阈值约 ≥ 2.0 V，低电平 ≤ 0.8 V。
  - **3.3 V CMOS/LVTTL**：高电平输出接近 V<sub>CC</sub>（如 ≥ 2.4 V @ 3.3 V），低电平接近 0 V；输入阈值随 V<sub>CC</sub> 缩放。
  - **VIH/VIL 与 VOH/VOL** 定义了驱动器与接收器能否可靠识别；跨电压域（5 V ↔ 3.3 V）直连可能损坏引脚或误判，需电平转换或容忍 5 V 输入的 3.3 V 器件。
- **对 wiki 的映射：** [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)

### 2) Texas Instruments — SN74 系列逻辑电平与总线收发

- **来源：** TI [SN74LVC 系列数据手册](https://www.ti.com/logic-circuit/overview.html)（代表 3.3 V LVC 逻辑）；典型器件如 SN74LVC1T45（双向电平转换）。
- **要点：**
  - 现代 MCU 多为 **3.3 V CMOS UART**；USB 转串口桥（CP2102、FT232 等）TTL 侧亦多为 3.3 V 或可通过跳线选择。
  - 板内 **TX/RX/GND** 三线即可工作；无硬件流控时 RTS/CTS 可悬空。
- **对 wiki 的映射：** [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)

### 3) Microsoft Learn — UART 架构（逻辑层与驱动模型）

- **来源：** [UART architecture](https://learn.microsoft.com/en-us/windows-hardware/drivers/serports/uart-architecture)（Windows 驱动架构参考，描述 UART 硬件抽象）。
- **要点：**
  - UART 负责 **波特率、数据位、校验、停止位** 的异步串行化；电气层由外部收发器或直连 TTL 实现。
  - 发送/接收 FIFO、中断与 DMA 是固件实时性设计的关键（printf 阻塞问题）。
- **对 wiki 的映射：** [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)、[uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

### 4) Wikipedia — UART 与异步串行

- **来源：** [Universal asynchronous receiver-transmitter](https://en.wikipedia.org/wiki/Universal_asynchronous_receiver-transmitter)
- **要点：**
  - 异步：无独立时钟线，双方约定波特率；帧结构为 **起始位 + 数据位 + 可选校验 + 停止位**。
  - 「TTL 串口」= UART 协议 + 单端 CMOS/TTL 电平，常见于 MCU 调试、IMU、GNSS、遥控接收机。
- **对 wiki 的映射：** [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)

## 推荐继续阅读（外部）

- NXP [AN11158 — Level shifting techniques in I2C-bus design](https://www.nxp.com/docs/en/application-note/AN11158.pdf)（电平转换通用思路，适用于 UART TTL 跨压）
- SparkFun [Logic Levels](https://learn.sparkfun.com/tutorials/logic-levels)（3.3 V / 5 V 工程直觉）

## 当前提炼状态

- [x] 摘要与 wiki 映射
