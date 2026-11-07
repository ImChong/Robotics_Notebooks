# EtherCAT Technology Group：EtherCAT 技术总览（官方）

> 来源归档（site / 行业组织官方技术页）

- **标题：** EtherCAT - the Ethernet Fieldbus（EtherCAT Technology Overview）
- **类型：** site
- **链接：** <https://www.ethercat.org/en/technology.html>；FAQ：<https://www.ethercat.org/en/faq.html>
- **机构：** EtherCAT Technology Group（ETG）；协议由 Beckhoff Automation 开发，2003 年 4 月发布，ETG 2003 年 11 月成立，协议公开于 IEC 61158
- **入库日期：** 2026-11-07
- **核对范围：** 技术总览页 1.1–1.12 与第 2–4 章正文（主站/从站实现、一致性认证、国际标准化）全部读完；FAQ 页只检索了与周期、轴数、从站间通信相关条目，未逐条读完；EtherCAT 规范本体（ETG.1000 系列）需会员，未读；页内 Read more 链接指向的子页未读。
- **通读：** 已完成
- **沉淀到 wiki：** [CAN vs EtherCAT 关节总线选型](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

## 核对摘录

- **设计目标：** 周期 ≤100 µs、同步抖动 ≤1 µs、低硬件成本；既适用硬实时也适用软实时。
- **工作原理（on the fly）：** 主站（MainDevice）是网段中唯一允许主动发帧的节点，帧依次经过各从站，从站在帧经过时读取寻址给自己的数据、同时插入自己的数据，延迟仅为硬件传播延迟；最后一个节点检测到开口端口后利用全双工把帧回送主站。有效数据率可超 90%。
- **主站只需标准以太网 MAC**，无需专用通信处理器；从站用 EtherCAT SubDevice Controller（ESC）在硬件中处理，网络性能与单个从站实现无关。
- **帧：** 载荷直接嵌入标准以太网帧（EtherType 0x88A4），不走 TCP/UDP；一帧含一个或多个 datagram；周期过程数据用**逻辑寻址**（4 GB 地址空间），可对驱动用短周期、对 I/O 用长周期；单帧经 DMA 收发，CPU 不必逐字节拷贝。
- **拓扑：** 线、树、星均可；100BASE-TX 站间最长 100 m；单网段最多 65,535 个设备；EtherCAT P 在同一根线上传数据与 24 V 供电。
- **分布式时钟（DC）：** 全硬件同步，传播延迟被测量并补偿，系统抖动显著低于 1 µs；采样/输出由从站本地时钟触发而非帧到达时刻，因此主站可在普通以太网硬件上以软件实现，几微秒量级的发送抖动不影响 DC 精度。
- **诊断：** 每个 ESC 用校验和检查帧，错误计数器可定位故障点；Working Counter 校验每个 datagram 的一致性。
- **冗余：** 线形拓扑末端接回主站第二网口即成环，恢复时间 <15 µs，节点无需改动。
- **CoE（CAN application protocol over EtherCAT）：** 提供与 CANopen（EN 50325-4）相同的对象字典、PDO 映射与 SDO 机制，网络管理也相似，便于把原有 CANopen 设备迁到 EtherCAT。
- **CiA 402：** 页面明确写「驱动 Profile CiA 402 等设备 Profile 也可复用于 EtherCAT」，CoE 还可取消经典 8 字节 PDO 限制。
- **主站/从站实现：** 主站唯一硬件要求是以太网口（DMA），按功能分 Class A / Class B 主站；配置依赖 ESI（从站 XML）与 ENI（网络信息）文件；从站 ESC 可用 ASIC/FPGA/微控制器集成，多数情况 8 位 MCU 即够。
- **安全：** Safety over EtherCAT（FSoE）依据 IEC 61508，适用至 SIL 3，采用黑通道原则；页面称机器人行业有用 SoE 做 8 kHz 闭环的安全运动控制应用。
- **FAQ：** 从站间通信有两种——拓扑相关的下游直通，以及经主站转发（需两个总线周期）。

## 未核对 / 不能由本页支持的陈述

- 本页**没有**给出「100 轴 250 µs」这类具体轴数–周期对应关系；wiki 中如出现，只能写作第三方/经验值并标注。
