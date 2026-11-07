# CiA CAN 知识库：经典 CAN 与高速物理层

> 来源归档（site / 行业标准科普）

- **标题：** CAN knowledge — 经典 CAN、高速传输与历史（CiA 摘录索引）
- **类型：** site
- **来源：** CAN in Automation (CiA) — [CAN knowledge](https://www.can-cia.org/can-knowledge/)
- **入库日期：** 2026-05-19
- **复核日期：** 2026-11-07（重新取得下列 CiA 页面全文）
- **核对页面：** [CAN CC](https://www.can-cia.org/can-knowledge/can-cc)、[CAN HS transmission](https://www.can-cia.org/can-knowledge/can-hs-transmission)、[Physical layer options](https://www.can-cia.org/can-knowledge/physical-layer-options)、[History of CAN technology](https://www.can-cia.org/can-knowledge/history-of-can-technology)、[Designing a CAN network](https://www.can-cia.org/can-knowledge/designing-a-can-network)
- **核对范围：** 以上 5 页正文逐段读完（含仲裁、帧格式、5 种错误检测、HS 电平与终端、物理层选项汇总、历史、网络设计建议）；页尾的 CAN Newsletter 文章列表只看标题，未读各篇；ISO 11898 系列为付费标准，未读。
- **通读：** 已完成
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

## 2026-11-07 全文复核补充（CiA 原文）

### CAN CC 数据链路层（CiA「CAN CC」页）

- **特性：** 多主（multi-drop）、广播、复杂错误检测 + 自动重发、独特的故障限制（fault confinement）以保证全网数据一致；数据链路层见 ISO 11898-1，一致性测试见 ISO 16845-1。
- **标识符即优先级：** 11 位（CBFF）或 29 位（CEFF）；**数值越小优先级越高**；竞争时按位仲裁，采用 wired-AND（显性覆盖隐性），输掉的节点自动变接收者、等总线空闲后重发。CiA 称这使过载时低延迟可保证。
- **帧结构：** SOF → 仲裁场（ID + RTR）→ 控制场（IDE + DLC）→ 数据场 **0–8 字节** → 15 位 CRC → ACK → EOF → 3 位间歇场。IDE 参与仲裁：同 ID 前 11 位时基本帧胜过扩展帧。扩展帧代价：延迟至少多 20 位时间、带宽多约 20%、CRC 的检错性能较低（15 位 CRC 多项式针对 ≤112 位帧长优化）。
- **五种错误检测：** CRC、帧检查（格式错误）、ACK 错误、位监测、位填充（连续 5 个相同位后插入补码位）；任一节点检出错误即发错误帧使全网同步感知；故障节点按错误状态机最终 bus-off 断开。

### CAN HS 物理层（CiA「CAN HS transmission」页）

- ISO 11898-2；上限 1 Mbit/s；隐性两线约 2.5 V，显性 CAN_H≈3.5 V、CAN_L≈1.5 V；显性差分 0.9–2.0 V，隐性 −1.0–0.5 V，0.5–0.9 V 不确定区。
- 速率–长度：理论 1 Mbit/s 约 40 m（实际因连接器、收发器、采样点更短）；超过约 1 km 压降使显性位检测不可靠。电缆延迟建议 ≤5 ns/m。
- 推荐线型拓扑、两端 120 Ω（或 2×60 Ω 分裂终端 + 接地电容）。低功耗与选择性唤醒（ISO 11898-5/-6）已并入 ISO 11898-2:2016。

### 物理层选项与网络设计

- PMA 选项：CAN HS（≤1 Mbit/s）；CAN FD 收发器（数据段 ≥2 Mbit/s）；CAN SIC（ISO 11898-2:2024，抑制振铃，物理层页称数据段限 5 Mbit/s）；CAN SIC XL（FAST 模式 PWM 编码，最高 20 Mbit/s）；ISO 11898-3 容错低速（125 kbit/s，新设计不推荐）；单线 CAN（不推荐）。各节点须支持相同比特率，采样点不一致会限制速率–长度比。
- 「Designing a CAN network」页：理论上不会「过载」，只会拖延低优先级帧；**平均总线负载经验上不超过 50%**（短时更高可接受）；速率能低则低（更稳健）；CAN SIC 在复杂混合拓扑下可到 8 Mbit/s（与物理层页的 5 Mbit/s 口径不同，以具体收发器与拓扑为准）；星形/混合拓扑限制速度–长度比，未端接支线越短越好；长网络建议光/变压器隔离（会缩短最大长度或速率）、双绞线、必要时屏蔽；一致性测试 ≠ 互操作性。

### 历史（CiA「History」页）

- 1983 年 Bosch 的 Uwe Kiencke 启动；1986 年 2 月 SAE 底特律大会发布；1987 年中 Intel 82526 为首个 CAN 控制器芯片；ISO 11898 于 1993 年 11 月发布，1995 年增补 29 位扩展帧；2003 年 ISO 11898 拆分修订为 -1（数据链路层）、-2（HS）、-3（容错 LS）。1992 年成立 CiA。
- 与本归档早先摘录（1986 年 SAE 发布、线束减少仅为副产品、HS 电平与 120 Ω 终端、≤5 ns/m）逐项对照，均与 CiA 原文一致，未发现需要纠正之处。
