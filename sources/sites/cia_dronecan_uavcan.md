# CiA / DroneCAN：无人机与机器人 CAN 高层协议

> 来源归档（site + 开源规范入口）

- **标题：** UAVCAN (Cyphal) and DroneCAN — CiA 知识库 + DroneCAN 官网
- **类型：** site
- **来源：**
  - CiA：<https://www.can-cia.org/can-knowledge/uavcan-and-dronecan/>
  - DroneCAN：<http://dronecan.org/>
  - DSDL 仓库：<https://github.com/dronecan/DSDL>
- **入库日期：** 2026-05-19
- **一句话说明：** 面向无人机与移动机器人的开源 CAN 应用层：ArduPilot/PX4 生态事实标准，支持经典 CAN 与 CAN FD，强调 DNA 动态节点 ID 与 DSDL 消息描述。

## 为什么值得保留

- **与腿式/臂式「研究机器人」不同赛道但同技术栈**：多旋翼与部分地面平台用 DroneCAN 连接 ESC、GPS、传感器；理解其传输层分段与 CRC 有助于阅读开源飞控代码。
- **CAN FD 路线图**：DroneCAN 规划支持 FDCAN 与更大帧，与 [can-fd](../../wiki/concepts/can-fd.md) 知识链衔接。

## 核心摘录（CiA）

### 1) UAVCAN → Cyphal 与 DroneCAN 分叉

- **要点：** 开源 **Cyphal**（原 UAVCAN v1）与 **DroneCAN**（延续 UAVCAN v0 路线）并存；DroneCAN 为 ArduPilot、PX4 与 CAN 外设通信的主协议，规范与实现均开源。
- **对 wiki 的映射：** [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)

### 2) 传输层特性

- **要点：** 支持 **CAN CC 与 CAN FD**；多段广播/确认通信，首段带 CRC，toggle 位防双发；设备上电后可立即工作，仅需周期性广播状态。
- **对 wiki 的映射：** [can-fd](../../wiki/concepts/can-fd.md)

## 核心摘录（dronecan.org）

### 3) 关键特性

- **要点：** DroneCAN v1 与旧版 UAVCAN 协议一致，工业已有大量兼容设备；具备 DSDL、**DNA（动态节点 ID 分配）**、C/C++/Python 绑定、GUI 诊断、ArduPilot/PX4 成熟实现；近期待办含 **FDCAN**、能力通告消息等。
- **对 wiki 的映射：** [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)

## 推荐继续阅读（外部）

- DroneCAN Specification（dronecan.github.io）
- ArduPilot CAN 文档

## 当前提炼状态

- [x] 摘要完成
