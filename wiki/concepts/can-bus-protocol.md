---
type: concept
tags: [middleware, hardware, protocol, can-bus, fieldbus, embedded, robotics]
status: complete
updated: 2026-11-07
related:
  - ./can-fd.md
  - ./ethercat-protocol.md
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ../entities/canfestival.md
  - ../comparisons/can-vs-ethercat-joint-bus.md
  - ../formalizations/control-loop-latency-modeling.md
  - ../concepts/processor-in-the-loop-sim2real.md
  - ../queries/real-time-control-middleware-guide.md
original:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_canopen_overview.md
sources:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_canopen_overview.md
  - ../../sources/sites/cia_dronecan_uavcan.md
  - ../../sources/sites/canfestival-org.md
  - ../../sources/repos/canfestival.md
  - ../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md
  - ../../sources/sites/jetson-linux-r392-welcome-and-can.md
summary: "经典 CAN（Controller Area Network）总线：多主、按位仲裁的差分串行现场总线，1 Mbit/s 与 8 字节载荷上限，广泛用于腿式/人形关节驱动与车载 ECU，应用层常见 CANopen 或 DroneCAN。"
---

# CAN 总线（经典 CAN / CAN 2.0）

**CAN（Controller Area Network）** 是一种 **多主、广播式、带硬件仲裁** 的串行现场总线。在机器人里，它最常见于 **主控板 ↔ 关节电机驱动器** 的反馈与力矩指令链路；在汽车里则是 ECU 组网的事实标准。本页以 CiA（CAN in Automation）官方知识库为一手依据，经典 CAN 的数据链路层见 ISO 11898-1，高速物理层见 ISO 11898-2。

## 一句话定义

多条节点共享一对差分线（CAN_H / CAN_L），任意节点可发起帧；若 ID 冲突，**数值越小（越 dominant）的标识符赢得总线**——无需中央主站即可安全共享介质。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAN | Controller Area Network | 本页主题：多主、按位仲裁的差分串行总线 |
| CAN CC / HS | CAN Classic / High-Speed | 经典 CAN 数据链路层（CC）与 ≤1 Mbit/s 高速物理层（HS） |
| CAN FD | CAN with Flexible Data-Rate | 载荷最长 64 字节、数据段可提速的扩展，见 [CAN FD](./can-fd.md) |
| CiA | CAN in Automation | CAN 用户与厂商组织，CANopen 规范的维护者 |
| ISO 11898 | ISO 11898 series | CAN 国际标准：-1 数据链路层，-2 高速物理层 |
| DLC | Data Length Code | 控制场内 4 位数据长度码，经典 CAN 数据场 0–8 字节 |
| CRC | Cyclic Redundancy Check | 15 位循环冗余校验，保护帧完整性 |
| ID | Identifier | 11 位（基本帧）或 29 位（扩展帧）标识符，同时决定优先级 |
| CANopen | CANopen | CiA 301 定义的 CAN 高层协议（对象字典 + PDO/SDO/NMT） |
| PDO / SDO | Process / Service Data Object | CANopen 的过程数据（周期实时）与服务数据（配置）通信对象 |
| EtherCAT | Ethernet for Control Automation Technology | 高实时性工业以太网总线，见 [EtherCAT](./ethercat-protocol.md) |
| DroneCAN | DroneCAN | ArduPilot / PX4 使用的开源 CAN 应用层 |

## 为什么重要

- **成本与生态**：收发器与 MCU 内置 CAN 控制器便宜，开源人形/四足 BOM 里常见 USB 转 CAN 适配器、板载 CAN 走线；Jetson 全系列也提供板载 CAN 控制器（见工程实践）。
- **优先级即延迟保证**：CiA 指出，访问总线的优先级由 ID 决定，传输请求按重要性处理，**过载时尤其有利**，可为实时系统保证高优先级报文的低延迟。
- **广播与易扩展**：每帧广播，只做接收的新节点可直接挂网而无需改动现有节点。
- **Sim2Real 隐形杀手**：真机失败常来自 **CAN 迟到、解析错误、周期错位**，而非仅摩擦系数——见 [处理器在环 Sim2Real](./processor-in-the-loop-sim2real.md)。

## 核心机制

### 1. 帧与仲裁

- **两种帧格式**：基本帧（CBFF）11 位 ID，扩展帧（CEFF）29 位 ID（11 位基本 ID + 18 位扩展）；数据场 **0–8 字节**；2003 年起所有 CAN CC 实现须同时支持两种格式。
- **帧结构**：SOF → 仲裁场（ID + RTR）→ 控制场（IDE + DLC）→ 数据场 → 15 位 CRC → ACK → EOF → 3 位间歇场。接收节点在 ACK 槽发显性位确认。
- **按位仲裁**：所有节点在采样点逐位比较；总线按 wired-AND，显性覆盖隐性。发隐性位却读到显性位的节点输掉仲裁、自动转为接收者，等总线空闲后重发——ID 数值越小优先级越高，且赢家的帧不被破坏（非破坏性仲裁）。IDE 位也参与仲裁，同样前 11 位 ID 时基本帧胜过扩展帧。
- **扩展帧的代价（CiA）**：延迟至少多 20 个位时间、占用带宽约多 20%、检错性能较低（15 位 CRC 多项式针对 ≤112 位的帧长优化）。
- **五种错误检测**：CRC、帧格式检查、ACK 错误、位监测（发送者回读）、位填充（连续 5 个相同位后插入补码位）。任一节点检出错误即发送 **错误帧** 中止当前传输，使全网同时感知，接收者随后期待重发；故障节点经错误状态机最终 **bus-off** 断开，避免长期拖垮总线。

### 2. 物理层（CAN HS）

- ISO 11898-2 **高速收发器**，上限 **1 Mbit/s**。隐性时两线均约 2.5 V；显性时 CAN_H≈3.5 V、CAN_L≈1.5 V；显性差分范围 0.9–2.0 V，隐性差分 −1.0–0.5 V，0.5–0.9 V 为不确定区。
- **速率–长度**：因为按位仲裁，速率越高允许网线越短；理论上 1 Mbit/s 约 40 m，实际因连接器、收发器和采样点设置更短；降速可拉长，但约 1 km 后压降使显性位不再可靠。线缆延迟建议 ≤5 ns/m。
- **拓扑与端接**：推荐线型拓扑，两端 120 Ω 终端匹配线缆阻抗（有时拆成 2×60 Ω 加接地电容）；星形与混合拓扑会限制速度–长度比，未端接的支线应尽量短。
- 收发器可带低功耗与选择性唤醒（并入 ISO 11898-2:2016）。更新的物理层选项（CAN FD 收发器、CAN SIC、CAN SIC XL）见 [CAN FD](./can-fd.md)。

### 3. 应用层：你实际在用的往往不止 CAN

| 协议 | 典型场景 |
|------|----------|
| **CANopen**（CiA 301 等） | 面向运动控制机械手起家；设备由协议栈 + 应用软件 + **对象字典**（16 位索引：1000h–1FFFh 通信参数，2000h–5FFFh 厂商私有，6000h–9FFFh 标准化参数）三部分组成；协议含 SDO、PDO、NMT、特殊功能与错误控制；比特率 10–1000 kbit/s；开源栈见 [CanFestival](../entities/canfestival.md) |
| **DroneCAN** | ArduPilot / PX4 与 CAN 外设（ESC、传感器）；推荐比特率 1 Mbit/s、500/250/125 kbit/s，采样点位于位时间的 75%；同时支持 CAN CC 与 CAN FD（CiA） |
| **厂商私有协议** | 部分人形/四足电机 SDK 的紧凑二进制帧 |
| **CoE** | 在 EtherCAT 上传 CANopen 对象，与 CANopen 共用对象字典、PDO、SDO，并可复用 CiA 402 驱动 Profile（见 [EtherCAT](./ethercat-protocol.md)；ETG 原文） |

CiA 402 在 CiA 的 CANopen 页只列了编号；其「驱动 Profile」含义出自 ETG 对 CoE 的描述。

## 工程实践

### 总线负载估算（推导，非 CiA 数字）

CiA 的经验法则是**平均总线负载不超过 50%**（短时更高可接受），并建议尽量少发无新信息的报文（CANopen 的 PDO 可灵活参数化以做到这点）。按标准帧各域位宽估算一帧 8 字节数据约 **111 位**（SOF 1 + ID 11 + RTR 1 + IDE 1 + 保留 1 + DLC 4 + 数据 64 + CRC 15 + CRC 界定 1 + ACK 2 + EOF 7 + 间歇 3，位填充另计；位宽取自规范常识，CiA 页面未逐域列出）。1 Mbit/s 下约 111 µs/帧，即**单条总线理论上限约 9000 帧/s，50% 负载约 4500 帧/s**（推导）。

例：12 个关节、每关节每周期 1 条指令帧 + 1 条反馈帧、周期 1 kHz，需要 24000 帧/s——已超出单条 1 Mbit/s 总线的物理上限（推导），因此实际多拆成多条 CAN、降低周期、改用 [CAN FD](./can-fd.md) 或 [EtherCAT](./ethercat-protocol.md)。

### 与人形分层总线的关系

经典 CAN 的 **8 B + 单速率** 硬伤，在多指传感、多轻载关节周期上报时会迫使频繁分包、抬高总线负载。量产人形叙事里更常见的是（出自公众号二手归档，**未经一手资料核实**）：

- **保留经典 CAN / CANopen** 于中等轴数科研机或已有电机生态；
- **末端升级 [CAN FD](./can-fd.md)**（灵巧手、轻载关节、分布式传感、BMS/STO）；
- **主干仍走 [EtherCAT](./ethercat-protocol.md)** 做大功率关节与视觉同步。

历史脉络（Bosch → ISO 11898 → CAN FD）见 [公众号归档](../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md)，已与 CiA 历史页核对：1986 年 SAE 发布、1987 年 Intel 82526 首款控制器、1993 年 ISO 11898、2003 年修订拆分为 -1/-2/-3。正式参数以 CiA / ISO 为准。

### 主控侧：Jetson 的板载 CAN

NVIDIA Jetson Linux 开发指南（r39.2）说明，Jetson Thor、AGX Orin、Orin NX、Orin Nano 均带 Time-Triggered CAN 控制器（`mttcan`）：支持标准与扩展帧，位速率可配置 10 kbps–1 Mbps，数据段 Thor 最高 8 Mbps、其余平台最高 15 Mbps；Orin NX/Nano 1 路、AGX Orin 与 Thor 各 2 路；除 Thor 开发套件自带收发器外，其余平台需外接 ≥3.3 V 的收发器。Linux 下加载 `can`、`can_raw`、`mttcan` 模块后用 `ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on` 之类命令启用（SocketCAN 接口）。详见 [NVIDIA Jetson](../entities/nvidia-jetson.md)。

## 局限与常见误区

- **「CAN 就是 1 Mbps 随便接」**：拓扑、终端、接地与采样点不一致都会让「理论上 1 M」在真机上变成间歇 NACK；CiA 明确指出速度与长度、采样点、收发器都相互制约。
- **「用 USB 转 CAN 就等同板载实时」**：USB 调度与批量传输会引入抖动，经验上毫秒量级，不适合单独承担 500 Hz–1 kHz 全关节闭环，宜板载 CAN + RT 线程（经验判断，推测，CiA 未给数字）。
- **混淆 CAN 与 UART/RS485**：UART 无硬件仲裁；RS485 半双工需方向控制，协议栈完全不同——见 [UART 串行通信](./uart-serial-communication.md)。
- **「经典 CAN 已够用末端传感」**：手指/多 IMU 周期上报时 8 B 分包常先撞负载墙，再谈换 EtherCAT——优先评估 [CAN FD](./can-fd.md)。
- **「一致性测试通过 = 互操作」**：CiA 提醒一致性测试像拼写检查，并不证明互操作；CANopen 另有互操作性测试与 plug fest。
- **适用边界**：经典 CAN 帧载荷 ≤8 字节、速率 ≤1 Mbit/s，扩展帧还要多付约 20% 带宽；总线利用率过高时低优先级报文会被无限拖延（不是「过载丢包」，而是饥饿）。
- **本页未覆盖**：CiA 402 Profile 细节、CANopen PDO 映射的具体配置、J1939/DeviceNet 等其他高层协议。

## 关联页面

- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md) — CANopen / 私有帧 / CiA 402 等应用层选型
- [CanFestival](../entities/canfestival.md) — 开源 ANSI-C CANopen Master/Slave 栈
- [CAN FD（Flexible Data Rate）](./can-fd.md)
- [EtherCAT 协议基础](./ethercat-protocol.md)
- [CAN vs EtherCAT：关节总线选型](../comparisons/can-vs-ethercat-joint-bus.md)
- [Hardware-in-the-Loop（硬件在环）](./hardware-in-the-loop.md) — 工业臂驱动 HIL 等 CAN 闭环台架案例
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [NVIDIA Jetson](../entities/nvidia-jetson.md) — 机载计算平台的板载 CAN

## 参考来源

- [CiA：经典 CAN、HS 物理层与历史](../../sources/sites/cia_can_knowledge_can_classic_and_hs.md)（原文，已通读）
- [CiA：CANopen 概览](../../sources/sites/cia_canopen_overview.md)（原文，已通读）
- [CiA / DroneCAN：无人机 CAN 应用层](../../sources/sites/cia_dronecan_uavcan.md)（CiA UAVCAN/DroneCAN 页已读，DroneCAN 官网未读）
- [CanFestival 官网](../../sources/sites/canfestival-org.md)
- [CanFestival 仓库归档](../../sources/repos/canfestival.md)
- [微信：CAN/CAN FD 到人形底层总线](../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md)（二手）
- [Jetson Linux r39.2 开发指南 Welcome 与 CAN 章节归档](../../sources/sites/jetson-linux-r392-welcome-and-can.md)（CAN 章节）

## 推荐继续阅读

- CiA [CAN knowledge](https://www.can-cia.org/can-knowledge/) 目录；其中 [CAN CC](https://www.can-cia.org/can-knowledge/can-cc) 与 [Designing a CAN network](https://www.can-cia.org/can-knowledge/designing-a-can-network) 最适合工程入门
- ISO 11898 系列（经典 CAN 数据链路与物理层）
- Jetson Linux 开发指南 [Controller Area Network (CAN)](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/HR/ControllerAreaNetworkCan.html)
- CanFestival：<https://canfestival.org/>
