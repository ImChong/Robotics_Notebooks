# CiA：CANopen 概览

> 来源归档（site）

- **标题：** CANopen CC – The standardized embedded network（CiA）
- **类型：** site
- **来源：** CAN in Automation (CiA)
- **链接：** https://www.can-cia.org/can-knowledge/canopen/
- **入库日期：** 2026-05-19
- **复核日期：** 2026-11-07（重新取得 CiA 页面全文）
- **核对页面：** [CANopen](https://www.can-cia.org/can-knowledge/canopen)
- **核对范围：** CANopen 页正文读完（设备三部分结构、对象字典索引分区、5 类协议、CiA 301 摘要、Profile 编号列表）；CiA 402 等 Profile 在该页仅列编号、无说明正文，因此 CiA 402「驱动与运动控制」的含义**未**由本页证实；CiA 规范本体（301 等）需登录下载，未读。
- **通读：** 已完成
- **一句话说明：** 基于 CAN 的嵌入式高层协议族：标准化 COB、设备/应用 Profile，实现「即插即用」式运动控制与医疗、轨交等现场网络。

## 为什么值得保留

- **机器人常见栈**：许多关节驱动器实现 **CANopen** 或 **CiA 402**（驱动与运动控制 Profile）；EtherCAT 应用层 **CoE** 即 CANopen over EtherCAT。
- **减轻底层负担**：开发者不必逐位处理验收滤波与位时序细节，转而配置对象字典与 PDO/SDO。

## 核心摘录

### 1) CANopen CC 与 CANopen FD

- **要点：** CANopen CC 基于经典 CAN；**CANopen FD** 基于 CAN FD。包含高层协议与 Profile 规范，配置高度灵活。
- **对 wiki 的映射：** [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)、[can-fd](../../wiki/concepts/can-fd.md)、[canfestival](../../wiki/entities/canfestival.md)

### 2) 通信对象与 Plug-and-play

- **要点：** 提供时间关键过程、配置与网络管理的标准化 **COB（communication objects）**；设备/接口/应用 Profile 支持互操作与互换，同时允许厂商扩展功能。
- **对 wiki 的映射：** [ethercat-protocol](../../wiki/concepts/ethercat-protocol.md)（CoE 章节）、[canfestival](../../wiki/entities/canfestival.md)

### 3) 典型应用领域

- **要点：** 最初面向运动控制机械手；现亦用于医疗设备、非道路车辆、海事、铁路、楼宇自动化等。
- **对 wiki 的映射：** [can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)、[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

## 推荐继续阅读（外部）

- CiA 402 驱动 Profile（运动控制设备）
- ETG：CANopen over EtherCAT (CoE)
- 开源栈：[CanFestival](https://canfestival.org/)（归档见 [canfestival-org.md](canfestival-org.md)）

## 当前提炼状态

- [x] 摘要完成

## 2026-11-07 全文复核补充（CiA 原文）

- CANopen CC 基于 CAN CC，CANopen FD 基于 CAN FD；最初面向运动控制机械手（handling systems），现用于医疗、非公路车辆、海事、铁路、楼宇自动化。
- **设备三部分：** 协议栈、应用软件、**对象字典**（16 位索引，4 位十六进制）。索引 1000h–1FFFh：通信参数；2000h–9FFFh：应用参数，其中 2000h–5FFFh 厂商私有、6000h–9FFFh 标准化。
- **CANopen CC 协议族：** SDO、PDO、NMT、特殊功能协议、错误控制协议。比特率 10–1000 kbit/s（CiA 301 规定位时序）；寻址以 11 位 ID 为主，亦支持 29 位；假设物理层为 ISO 11898-2 但不排除其他。
- CiA 301 v4.2.0（2011-02-21，PAS）：应用层与通信子协议（数据类型、编码、对象字典、通信服务、NMT、预定义 COB-ID 连接集、Emergency/Time-stamp/Sync 内容）。
