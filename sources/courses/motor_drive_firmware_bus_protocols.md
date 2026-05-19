# 电机驱动器底软通信协议（课程式索引）

> 来源归档（course / 工程选型合集）

- **标题：** 主控 ↔ 关节驱动器底软：现场总线与应用层协议索引
- **类型：** course
- **来源：**
  - CiA：[CANopen](https://www.can-cia.org/can-knowledge/canopen/)、[CiA 402 驱动与运动控制 Profile](https://www.can-cia.org/can-knowledge/canopen/cia-402/)（页面需登录下载部分 PDF，公开页可索引）
  - CiA：前述 CAN / CAN FD / 物理层条目（见 `sources/sites/cia_can_*.md`）
  - DroneCAN：<http://dronecan.org/>、<https://github.com/dronecan/DSDL>
  - 学术四足常见「MIT 协议」公开实现线索：MIT Cheetah Software、各开源 quadruped 仓库 README 中的 motor command 结构说明（以具体仓库文档为准，本页不固化字节表）
- **入库日期：** 2026-05-19
- **一句话说明：** 把「电机底软通信」拆成物理总线 + 应用协议 + 控制语义三层，汇总 CANopen/CiA402、CoE、厂商私有、DroneCAN、MIT 紧凑帧、Modbus 等种类的适用场景与优缺点，供机器人主控集成选型。

## 为什么值得保留

- 上一批 ingest 已覆盖 **CAN/UART 物理层**，但读者真正困惑的是：**同一种 CAN 线上跑的是 CANopen 还是厂商私有？和 EtherCAT CoE 什么关系？**
- 腿式/人形集成时，**底软协议选错**会导致：仿真里 PD 增益与真机对象字典不一致、周期错位、或无法做处理器在环。

## 核心摘录

### 1) 三层模型

- **要点：** ① **物理/链路**（CAN、CAN FD、EtherCAT、UART/RS485）；② **应用/传输**（CANopen、CoE、DroneCAN、私有帧、Modbus）；③ **控制语义**（CSP/CSV/CST、阻抗/PD 目标、力矩直给）。选型必须三层对齐。
- **对 wiki 的映射：** [motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

### 2) CANopen + CiA 402

- **要点：** CANopen 提供对象字典、PDO/SDO、NMT；**CiA 402** 标准化伺服驱动状态机与模式（位置/速度/力矩等）。工业机器人与部分关节模组默认 profile；EtherCAT 上 **CoE** 复用同一对象模型。
- **对 wiki 的映射：** 同上 + [can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)、[ethercat-protocol](../../wiki/concepts/ethercat-protocol.md)

### 3) 厂商私有 CAN 与 MIT 紧凑帧

- **要点：** 消费级/科研电机常为 **固定 ID + 紧凑 struct**（常称 MIT/Cheetah 风格）：带宽高、文档随厂商变；**优点**是易写、低延迟；**缺点**是互操作性差、Sim2Real 需绑定 SDK。Unitree、达妙、宇树等各有协议族，应以官方 SDK 为准。
- **对 wiki 的映射：** [motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

### 4) DroneCAN

- **要点：** 无人机 ESC/传感器生态；DSDL 描述消息；支持 CAN FD；与关节伺服 CANopen 栈不同，但同属「CAN 上的应用层」选型。
- **对 wiki 的映射：** [cia_dronecan_uavcan.md](../sites/cia_dronecan_uavcan.md)

## 推荐继续阅读（外部）

- CiA 402 官方页与 CANopen 设备 Profile 列表
- ETG：CANopen over EtherCAT (CoE) 应用指南

## 当前提炼状态

- [x] 摘要与 wiki 总览页映射
