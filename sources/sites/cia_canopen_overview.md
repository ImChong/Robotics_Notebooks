# CiA：CANopen 概览

> 来源归档（site）

- **标题：** CANopen CC – The standardized embedded network（CiA）
- **类型：** site
- **来源：** CAN in Automation (CiA)
- **链接：** https://www.can-cia.org/can-knowledge/canopen/
- **入库日期：** 2026-05-19
- **一句话说明：** 基于 CAN 的嵌入式高层协议族：标准化 COB、设备/应用 Profile，实现「即插即用」式运动控制与医疗、轨交等现场网络。

## 为什么值得保留

- **机器人常见栈**：许多关节驱动器实现 **CANopen** 或 **CiA 402**（驱动与运动控制 Profile）；EtherCAT 应用层 **CoE** 即 CANopen over EtherCAT。
- **减轻底层负担**：开发者不必逐位处理验收滤波与位时序细节，转而配置对象字典与 PDO/SDO。

## 核心摘录

### 1) CANopen CC 与 CANopen FD

- **要点：** CANopen CC 基于经典 CAN；**CANopen FD** 基于 CAN FD。包含高层协议与 Profile 规范，配置高度灵活。
- **对 wiki 的映射：** [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)、[can-fd](../../wiki/concepts/can-fd.md)

### 2) 通信对象与 Plug-and-play

- **要点：** 提供时间关键过程、配置与网络管理的标准化 **COB（communication objects）**；设备/接口/应用 Profile 支持互操作与互换，同时允许厂商扩展功能。
- **对 wiki 的映射：** [ethercat-protocol](../../wiki/concepts/ethercat-protocol.md)（CoE 章节）

### 3) 典型应用领域

- **要点：** 最初面向运动控制机械手；现亦用于医疗设备、非道路车辆、海事、铁路、楼宇自动化等。
- **对 wiki 的映射：** [can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

## 推荐继续阅读（外部）

- CiA 402 驱动 Profile（运动控制设备）
- ETG：CANopen over EtherCAT (CoE)

## 当前提炼状态

- [x] 摘要完成
