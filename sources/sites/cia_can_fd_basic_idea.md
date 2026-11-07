# CiA：CAN FD（Flexible Data Rate）基本思想

> 来源归档（site）

- **标题：** CAN FD: The basic idea（CiA）
- **类型：** site
- **来源：** CAN in Automation (CiA)
- **链接：** https://www.can-cia.org/can-knowledge/can-fd-the-basic-idea/
- **入库日期：** 2026-05-19
- **复核日期：** 2026-11-07（重新取得 CiA 页面全文）
- **核对页面：** [CAN FD: the basic idea](https://www.can-cia.org/can-knowledge/can-fd-the-basic-idea)、[CAN FD light](https://www.can-cia.org/can-knowledge/can-fd-light)、[CANopen FD](https://www.can-cia.org/can-knowledge/canopen-fd-the-art-of-embedded-networking)
- **核对范围：** 三页正文读完（含帧结构图表的字段说明、DLC 扩展表、FD 帧格式、FD light、CANopen FD 的应用层变化）；ISO 11898-1:2024 付费，未读；CiA 下方的 Newsletter 列表只看标题。
- **通读：** 已完成
- **一句话说明：** Bosch 2011 年起与车企推动的 CAN FD 扩展：在仲裁段保持 CAN 兼容速率，在数据段提高比特率并将载荷从 8 字节扩至 64 字节。

## 为什么值得保留

- **解决经典 CAN 两大瓶颈**：① 净荷仅 8 byte；② 有效吞吐受 1 Mbit/s 限制。
- **机器人场景**：多关节状态/力矩帧更宽、更高刷新需求时，CAN FD 已成新驱动器与 USB2CAN 适配器常见选项。
- **与 DroneCAN / CANopen FD 衔接**：应用层协议在 CAN FD 物理层上继续演进。

## 核心摘录

### 1) 双速率与基本思路

- **要点：** CAN FD 数据帧可在 **仲裁阶段** 与 **数据阶段** 使用不同比特率；仲裁阶段速率受拓扑限制（≤1 Mbit/s）；数据阶段受收发器能力限制可更高。核心思想：**仅一个节点发送时可提速**，ACK 前需重新同步各节点。
- **对 wiki 的映射：** [can-fd](../../wiki/concepts/can-fd.md)

### 2) FDF / BRS 位与帧结构

- **要点：** 原保留位之一用作 **FDF（FD frame）位**：隐性表示 CAN FD 帧，显性表示经典 CAN CC 帧。**BRS（bit rate switch）**：隐性时在数据阶段启用第二比特率，显性则全程用仲裁段时序。帧字段与经典 CAN 同名（SOF、ID、控制、数据、CRC、ACK、EOF 等），但 CRC 与控制域扩展。
- **对 wiki 的映射：** [can-fd](../../wiki/concepts/can-fd.md)、[can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)

### 3) 吞吐提升量级

- **要点：** 更大载荷提高协议效率；仲裁与数据阶段比特率比约 **1:8** 时，考虑 FD 帧头与 CRC 变长，整体吞吐约可达经典 CAN 的 **~6×**（CiA 表述为近似值，依帧长与配置而异）。
- **对 wiki 的映射：** [can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

## 推荐继续阅读（外部）

- CiA：[CANopen FD – The art of embedded networking](https://www.can-cia.org/can-knowledge/canopen-fd-the-art-of-embedded-networking/)
- CiA：[CAN FD light](https://www.can-cia.org/can-knowledge/can-fd-light)

## 当前提炼状态

- [x] 摘要与 wiki 映射完成

## 2026-11-07 全文复核补充（CiA 原文）

- 2011 年 Bosch 与车企合作启动 CAN FD；解决两个限制：1 Mbit/s 与 8 字节。**仲裁段速率受拓扑限制（≤1 Mbit/s），数据段速率受收发器特性限制**；仅一个节点发送时可提速，ACK 槽前需重新同步。
- 吞吐：仲裁:数据比特率 1:8 时，考虑 FD 帧更长的控制场与 CRC，吞吐约为经典 CAN 的 **6 倍**（CiA 原话 approximately six times）。
- FDF 位（隐性=FD 帧）、BRS 位（隐性=切换到数据段比特率，在 BRS 采样点切换）、ESI 位（显性=错误主动）；FD 帧格式 FBFF（11 位）/ FEFF（29 位）；**FD 不支持远程请求帧**；CAN CC 与 CAN FD 一并标准化于 ISO 11898-1:2024；FD 控制器同时支持 CC 帧。
- 与 CAN SIC 等收发器的配合见 [cia_can_knowledge_can_classic_and_hs.md](cia_can_knowledge_can_classic_and_hs.md)。
