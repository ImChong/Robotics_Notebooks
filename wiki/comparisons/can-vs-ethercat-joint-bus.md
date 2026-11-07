---
type: comparison
tags: [hardware, fieldbus, can-bus, can-fd, ethercat, uart, rs485, realtime, robotics]
status: complete
updated: 2026-11-07
related:
  - ../overview/motor-drive-firmware-bus-protocols.md
  - ../concepts/can-bus-protocol.md
  - ../concepts/can-fd.md
  - ../concepts/ethercat-protocol.md
  - ../concepts/uart-serial-communication.md
  - ../concepts/rs-485-serial-bus.md
  - ../entities/canfestival.md
  - ../entities/soem.md
  - ../comparisons/ethercat-vs-ethernet-ip.md
  - ../formalizations/control-loop-latency-modeling.md
original:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_can_fd_basic_idea.md
  - ../../sources/sites/ethercat-technology-group-technology.md
  - ../../sources/sites/ti-rs485-slla070d.md
sources:
  - ../../sources/sites/cia_can_knowledge_can_classic_and_hs.md
  - ../../sources/sites/cia_can_fd_basic_idea.md
  - ../../sources/sites/canfestival-org.md
  - ../../sources/repos/soem.md
  - ../../sources/courses/uart_rs485_serial_embedded.md
  - ../../sources/sites/ethercat-technology-group-technology.md
  - ../../sources/sites/ti-rs485-slla070d.md
  - ../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md
summary: "腿式与人形关节反馈选型：经典 CAN/CAN FD、EtherCAT 与 UART/RS485 在带宽、拓扑、成本、生态与硬实时上的对照；人形常见「EtherCAT 主干 + CAN FD 末端分支」分层读法。"
---

# CAN / CAN FD vs EtherCAT vs UART·RS485（关节与现场总线选型）

> 对比轴：**多轴硬实时关节环**、**中等轴数成本敏感平台**、**调试与外设** 三类需求。一手依据：CiA 知识库（CAN CC / HS / FD）、ETG 官方 EtherCAT 技术总览、TI RS-485 应用笔记；带「（推导）」「（推测）」的数字不是原文给出。

## 一句话定义

三类总线解决的是不同问题：**CAN** 是带硬件仲裁的小帧多主总线，**EtherCAT** 是单主站、帧在从站间「飞行处理」的工业以太网，**UART / RS-485** 只规定电气或异步字节流、不规定仲裁与帧语义。

- **≥20 轴、250 µs–1 ms 级同步**：优先 **EtherCAT**（或同等工业以太网现场总线）。（经验阈值，推测；依据见下文推导）
- **中等轴数、已有 CAN 电机生态、成本敏感**：**经典 CAN** 或 **CAN FD** + 合理分包仍够用。
- **日志、遥控、单传感器、产测**：**UART / RS485**；不要承担全关节 1 kHz 闭环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CAN | Controller Area Network | 多主差分总线，ID 即优先级、按位仲裁；经典载荷 ≤8 B |
| CAN FD | CAN with Flexible Data-Rate | 数据段可提速、载荷 ≤64 B；FD 帧不支持远程请求帧 |
| EtherCAT | Ethernet for Control Automation Technology | 单主站、帧飞行处理的实时工业以太网（IEC 61158） |
| ESC | EtherCAT SubDevice Controller | 从站硬件控制器，帧经过时在硬件中读写数据 |
| DC | Distributed Clocks | EtherCAT 分布式时钟，全硬件同步，抖动远小于 1 µs（ETG） |
| CoE | CAN application protocol over EtherCAT | 在 EtherCAT 上复用 CANopen 的对象字典 / PDO / SDO |
| PDO | Process Data Object | 周期实时过程数据通信对象 |
| UART | Universal Asynchronous Receiver-Transmitter | 异步串行外设，只有起止位界定字节 |
| RS-485 | TIA/EIA-485 | 差分多点电气层标准，最多 32 单位负载，不定义协议 |
| U.L. | Unit Load | RS-485 单位负载，总线上限 32 U.L. |
| ESI / ENI | EtherCAT SubDevice / Network Information | 从站描述 XML / 主站网络配置文件 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 为什么重要

关节总线决定了 **控制频率 × 轴数 × 每轴字节数** 能否装进一个控制周期，也决定多轴相位是否对齐。选错总线的后果不是「慢一点」，而是高轴数时低优先级报文饥饿、力矩指令迟到，真机表现为抖动或失稳（见 [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)）。

## 核心原理：核心对比表

| 维度 | 经典 CAN | CAN FD | EtherCAT | UART / RS485 |
|------|----------|--------|----------|----------------|
| 物理介质 | CAN_H/L 差分 | 同左，FD 收发器 | 100BASE-TX 以太网（站间 ≤100 m），也可光纤 | TTL / ±RS-232 / RS-485 差分 |
| 典型速率 | ≤ 1 Mbit/s（CiA） | 仲裁段 ≤1 Mbit/s；数据段 ≥2 Mbit/s，具体取决于收发器与拓扑（CiA 各页对 SIC 的上限写 5 或 8 Mbit/s） | 100 Mbit/s，有效数据率可超 90%（ETG） | RS-485 理论 >10 Mbps、最长约 1200 m（TI，二者不同时取到）；UART 波特率由收发双方约定 |
| 单帧载荷 | ≤ 8 B | ≤ 64 B | 标准以太网帧内含一个或多个 datagram | 流式字节 |
| 多节点仲裁 | 硬件按 ID（非破坏性） | 同左（仲裁段） | 无仲裁：只有主站能主动发帧 | 无（RS-485 一个时刻只允许一个驱动器，需上层主从协议） |
| 多轴时间同步 | 无硬件机制（软件时间戳，需上层如 TTCAN） | 同左 | **DC：全硬件，抖动远小于 1 µs**（ETG） | 通常无 |
| 周期能力 | 受 1 Mbit/s 与帧长限制，见下文推导 | 明显改善 | 设计目标周期 ≤100 µs、同步抖动 ≤1 µs（ETG）；「100 轴 250 µs」未在 ETG 页面找到，仅作第三方说法 | 不适用 |
| 线缆/拓扑 | 线型 + 两端 120 Ω | 同左，更严（CiA 建议线型） | 线 / 树 / 星；单网段最多 65,535 设备 | RS-485 总线（菊花链 + 短支线）+ 两端端接 |
| 机器人常见度 | 四足/人形电机 | 新驱动器 | 人形高端、工业臂 | 调试、IMU、遥控 |
| 应用层 | CANopen、DroneCAN、私有 | CANopen FD | CoE（CANopen 机制）、SoE、EoE、FoE、AoE | Modbus RTU、自定义 |

开源 CANopen 主/从站实现可参考 [CanFestival](../entities/canfestival.md)；EtherCAT 侧用户态主站见 [SOEM](../entities/soem.md)（与 IgH 对照见 [主站优化指南](../queries/ethercat-master-optimization.md)）。

### 带宽量级对比（推导）

以 20 个关节、每周期每轴 8 B 指令 + 8 B 反馈为例（推导，帧开销按标准规范常识）：

- **经典 CAN**：每条报文一帧，8 字节数据帧约 111 位（见 [CAN 总线](../concepts/can-bus-protocol.md) 的位宽拆解），1 Mbit/s 下约 111 µs；40 帧共约 4.4 ms，远超 1 ms 周期，且超过 CiA 建议的 50% 平均负载。
- **EtherCAT**：320 B 过程数据装进一个帧，加上以太网头、EtherCAT 头、FCS 与前导/帧间隙，线上约 372 B ≈ 30 µs（100 Mbit/s）；另加各从站的硬件转发延迟。线上时间相差约 150 倍（两个数量级），原因是 CAN 每个字节都要背帧头/CRC/ACK 开销，EtherCAT 把所有轴合并成一帧并有效数据率超 90%。
- **CAN FD**：载荷 64 B 可把多个轴的数据合并，CiA 给出仲裁:数据比特率 1:8 时吞吐约为经典 CAN 的 6 倍；是否够用取决于轴数与周期，需用上述方法逐项核算。

## 工程实践

### 何时从 CAN 升级到 EtherCAT

- 控制频率 × 轴数 × 每轴反馈字节数 **逼近 CiA 建议的平均总线负载 50%** 且延迟长尾可见。
- 需要 **分布式时钟** 做相位对齐力矩（步行、全身 WBC）。
- 已有 EtherCAT 驱动器供应链，主站栈（[SOEM](../entities/soem.md)、IgH 等）可接受。ETG 指出主站只需一个标准以太网口（通常经 DMA），主站分 Class A / Class B，配置依赖 ESI / ENI 文件。
- 想保留 CANopen 软件资产：CoE 提供相同的对象字典、PDO、SDO，CiA 402 等 Profile 可复用（ETG）。

### 何时 CAN FD 而非换 EtherCAT

- 现有 **CAN 布线/驱动器** 可固件升级 FD，希望 **减少换栈成本**。
- 轴数中等（例如 ≤12–20，经验阈值，推测），FD 组帧后总线裕量足够。
- **人形末端域**：灵巧手、腕/颈/踝轻载关节、足底力/IMU 等无图像大数据、腔体放不下 ESC——用 CAN FD 分支，主干仍 EtherCAT（见下节）。

### 人形分层读法：EtherCAT 主干 + CAN FD 分支

不必把整机做成「纯 CAN」或「纯 EtherCAT」二选一。第三方综述（公众号，二手，未经一手核实）归纳的量产人形常见分层：

| 域 | 总线 | 典型负载 |
|----|------|----------|
| 主干 | EtherCAT | 髋/膝大功率关节、视觉/图像 |
| 末端分支 | CAN FD | 五指手、轻载关节、分布式传感、BMS/STO、诊断 OTA |

相对 TTL 串口：共线 + 硬件仲裁 + 差分抗干扰；相对经典 CAN：64 B 少分包；相对全机 EtherCAT：末端省 ESC 与布线复杂度。细节与可信度边界见 [CAN FD 工程实践](../concepts/can-fd.md) 与 [公众号归档](../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md)。

### UART/RS485 的正确定位

- **不要**与 CAN/EtherCAT 争关节闭环带宽。
- **要**用于：USB 串口调试、无线遥控接收、单点 Modbus 传感器、产测命令。
- 若用 RS-485：TI 笔记强调需两端端接、短支线、故障保护偏置避免总线空闲误码；多驱动器同时使能会造成 line contention，必须由上层协议（主从轮询）避免。

## 局限与常见误判

| 误判 | 实际情况 |
|------|----------|
| USB2CAN 等于板载 CAN 实时性 | USB 主机调度会引入抖动（经验量级毫秒，推测），宜仅用于标定与非实时命令 |
| CAN FD 自动解决所有轴数问题 | 仲裁段仍 ≤1 Mbit/s，极高轴数、需要硬件同步时仍可能要 EtherCAT |
| 人形只能二选一总线 | 常见是 EtherCAT 主干 + CAN FD 末端分支，按域切分（二手说法） |
| RS485 拉长即可替代 CAN | 无硬件仲裁与标准化错误处理，协议设计负担重（TI：RS-485 只规定电气特性） |
| EtherCAT 主站就是「随便一台 PC」 | 主站只需以太网口，但周期抖动由主机调度决定；DC 容忍几微秒发送抖动，但仍需实时线程（见 [主站优化指南](../queries/ethercat-master-optimization.md)） |

其他局限：本页没有实测延迟数据；表中带「（推导）」的是基于规范位宽的估算，不替代对具体驱动器固件和主站栈的实测；CAN SIC / CAN XL 等新物理层本页只点到为止。

## 关联页面

- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md)
- [CanFestival](../entities/canfestival.md) — 开源 CANopen 栈（CAN 侧主/从对照）
- [SOEM](../entities/soem.md) — 开源 EtherCAT 用户态主站
- [CAN 总线](./../concepts/can-bus-protocol.md)
- [CAN FD](./../concepts/can-fd.md)
- [EtherCAT 协议基础](./../concepts/ethercat-protocol.md)
- [UART 串行通信](./../concepts/uart-serial-communication.md)
- [RS-485 串行总线](./../concepts/rs-485-serial-bus.md)
- [EtherCAT vs EtherNet/IP](./ethercat-vs-ethernet-ip.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)

## 参考来源

- [CiA：经典 CAN 与物理层](../../sources/sites/cia_can_knowledge_can_classic_and_hs.md)（原文，已通读）
- [CiA：CAN FD](../../sources/sites/cia_can_fd_basic_idea.md)（原文，已通读）
- [ETG：EtherCAT 技术总览](../../sources/sites/ethercat-technology-group-technology.md)（原文，已通读）
- [TI：RS-422 与 RS-485 标准概览 SLLA070D](../../sources/sites/ti-rs485-slla070d.md)（原文，已通读）
- [CanFestival 官网](../../sources/sites/canfestival-org.md)
- [SOEM 仓库归档](../../sources/repos/soem.md)
- [UART / RS-485 入门](../../sources/courses/uart_rs485_serial_embedded.md)（已发现其 SLLA383 链接指向 LIN 文档，见该归档更正）
- [微信：CAN/CAN FD 到人形底层总线](../../sources/blogs/wechat_ai_explore_yao_can_canfd_humanoid_bus.md)（二手）

## 推荐继续阅读

- [EtherCAT vs EtherNet/IP](./ethercat-vs-ethernet-ip.md)
- ETG [EtherCAT 技术总览](https://www.ethercat.org/en/technology.html)
- CiA [Designing a CAN network](https://www.can-cia.org/can-knowledge/designing-a-can-network/)
- TI [SLLA070D](https://www.ti.com/lit/an/slla070d/slla070d.pdf)
- CanFestival：<https://canfestival.org/>
