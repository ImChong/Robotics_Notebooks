# CiA CAN 知识库：经典 CAN 与高速物理层

> 来源归档（site / 行业标准科普）

- **标题：** CAN knowledge — 经典 CAN、高速传输与历史（CiA 摘录索引）
- **类型：** site
- **来源：** CAN in Automation (CiA) — [CAN knowledge](https://www.can-cia.org/can-knowledge/)
- **入库日期：** 2026-05-19
- **一句话说明：** CiA 对 ISO 11898 经典 CAN、HS 收发器电平、仲裁与线长–速率关系的公开说明，是机器人关节总线入门的一手索引。

## 为什么值得保留

- **行业标准入口**：CAN 由 Bosch 1986 年提出，现广泛用于汽车、工业机械、无人机与开源人形/四足底层驱动。
- **与机器人落地直接相关**：腿式/人形「主控 ↔ 关节驱动器」大量采用 CAN（或 CAN FD）；理解仲裁、1 Mbit/s 上限与拓扑约束，才能解释真机抖动与丢帧。
- **可与 EtherCAT、UART/RS485 对照**：本仓库已有 [EtherCAT 概念页](../../wiki/concepts/ethercat-protocol.md)，补 CAN 经典层形成现场总线选型链。

## 核心摘录

### 1) 历史与定位

- **要点：** 1986 年 SAE 底特律大会 Bosch 发布 Controller Area Network；最初目标是为汽车 ECU 增加功能，线束减少是副产品。今日乘用车几乎标配 CAN；列车、船舶、工厂自动化亦广泛采用。
- **对 wiki 的映射：**
  - [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)
  - [ethercat-protocol](../../wiki/concepts/ethercat-protocol.md)（CoE = CANopen over EtherCAT）

**来源页：** [History of CAN technology](https://www.can-cia.org/can-knowledge/history-of-can-technology)

### 2) CAN HS 物理层与 1 Mbit/s 约束

- **要点：** ISO 11898-2 定义 CAN HS（high-speed）收发器；总线差分线 CAN_H / CAN_L，隐性位约 2.5 V，显性位 CAN_H≈3.5 V、CAN_L≈1.5 V。经典 CAN 位速率上限 **1 Mbit/s**；因 bitwise 仲裁，**速率越高、允许网线越短**（理论 1 Mbit/s 约 40 m，实际因连接器与采样点更短）。推荐 **线型拓扑 + 两端 120 Ω 终端**；线缆传播延迟建议 ≤ 5 ns/m。
- **对 wiki 的映射：**
  - [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)
  - [control-loop-latency-modeling](../../wiki/formalizations/control-loop-latency-modeling.md)

**来源页：** [CAN HS transmission](https://www.can-cia.org/can-knowledge/can-hs-transmission)

### 3) 物理层选项总览（CAN FD / SIC / XL）

- **要点：** 常见 PMA 实现于收发器芯片：CAN HS、**CAN FD**（数据段更高比特率）、CAN SIC（抑制振铃）、CAN XL 等；节点须支持相同速率，采样点不一致会限制 speed/length。
- **对 wiki 的映射：**
  - [can-fd](../../wiki/concepts/can-fd.md)
  - [can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

**来源页：** [Physical layer options](https://www.can-cia.org/can-knowledge/physical-layer-options)

## 推荐继续阅读（外部）

- CiA：[CAN knowledge 目录](https://www.can-cia.org/can-knowledge/)
- ISO 11898 系列（需购买）：经典 CAN 与 CAN FD 数据链路层

## 当前提炼状态

- [x] 基础摘要与 wiki 映射
- [x] 对应 wiki 概念页 `can-bus-protocol.md` 已规划互链
