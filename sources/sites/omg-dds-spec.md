# OMG DDS / DDSI-RTPS 规范（一手标准）

> 来源归档

- **标题：** OMG Data Distribution Service (DDS) 1.4 · DDS Interoperability Wire Protocol (DDSI-RTPS) 2.5
- **类型：** site（国际标准 / 规范入口）
- **来源：** Object Management Group（OMG）
- **链接：**
  - DDS 1.4：https://www.omg.org/spec/DDS/1.4
  - DDSI-RTPS 2.5：https://www.omg.org/spec/DDSI-RTPS/2.5（系列入口：https://www.omg.org/spec/DDSI-RTPS/）
- **规范 PDF：**
  - DDS 1.4：`formal/15-04-10`（[PDF](https://www.omg.org/spec/DDS/1.4/PDF)）
  - DDSI-RTPS 2.5：`formal/22-04-01`（[PDF](https://www.omg.org/spec/DDSI-RTPS/2.5/PDF)）
- **入库日期：** 2026-07-28
- **一句话说明：** ROS 2 / Fast DDS / Cyclone DDS 所实现的 **DCPS API 语义** 与 **互操作线协议** 的权威定义；调 QoS、排互通问题应以本规范为最终依据。
- **沉淀到 wiki：** 是 → [`wiki/concepts/dds-communication.md`](../../wiki/concepts/dds-communication.md)

## 为什么值得保留

- 本库此前 DDS 页主要依赖 ROS 2 文档与二手叙述；OMG 规范页是 **标准层一手入口**。
- 区分两层：**DDS（DCPS + QoS）** vs **DDSI-RTPS（线上字节与发现）**——厂商文档常混谈，选型/互通调试需拆开。
- 机器可读 IDL（`dds_dcps.idl` 等）便于对照实现与 Unitree 等厂商自定义 IDL 消息。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 规范文本 | **公开可读**（OMG 站点提供 PDF / IDL；版权归 OMG，非软件开源许可） |
| 参考实现 | 不在本页；见 [Fast DDS](../repos/fast-dds.md)、[Cyclone DDS](../repos/cyclonedds.md) |

> 标准本身不是代码仓；「开源状态」指规范可获取性，实现开源见各 vendor 仓。

## DDS 1.4（2015-03）核心定位

| 概念 | 说明 |
|------|------|
| **DCPS** | Data-Centric Publish-Subscribe：Topic + Type + QoS 的数据中心化模型 |
| **实体** | DomainParticipant、Publisher/Subscriber、DataWriter/DataReader、Topic |
| **QoS** | Reliability、History、Durability、Deadline、Liveliness、Ownership 等策略契约 |
| **域隔离** | 仅同 Domain 内实体可发现与匹配 |
| **附带 IDL** | `dds_dcps.idl`、`dds_dlrl.idl`（规范机读附件） |

历史正式版：1.0（2004）→ 1.1 → 1.2 → **1.4（现行常用引用）**。

## DDSI-RTPS 2.5（2022-04）核心定位

| 概念 | 说明 |
|------|------|
| **互操作线协议** | 不同 DDS 实现互通的字节级协议 |
| **常见承载** | UDP/IP（亦可 TCP 等，视实现） |
| **与 DDS 映射** | RTPS Writer/Reader ↔ DDS DataWriter/DataReader |
| **发现** | SPDP/SEDP 等内置发现；组播失效时需配置 peers / Discovery Server |

历史正式版：2.0（2008）→ … → **2.5（现行常用引用）**。

## 相关 OMG 配套（实现覆盖度不一）

| 规范 | 用途 | 实现侧备注 |
|------|------|------------|
| [DDS-SECURITY](https://www.omg.org/spec/DDS-SECURITY/) | 认证 / 授权 / 加密 | Fast DDS、Cyclone 均声明支持插件 |
| [DDS-XTypes](https://www.omg.org/spec/DDS-XTypes/) | 结构化类型与演化 | Cyclone README 标明部分 caveats |
| [DDS-PSM-Cxx](https://www.omg.org/spec/DDS-PSM-Cxx/) | 标准 C++ API 映射 | Cyclone 有独立 `cyclonedds-cxx` 仓 |

## 对 wiki 的映射

- 主概念：[dds-communication](../../wiki/concepts/dds-communication.md)
- 实现实体：[fast-dds](../../wiki/entities/fast-dds.md)、[cyclone-dds](../../wiki/entities/cyclone-dds.md)
- ROS 2 栈：[ros2-basics](../../wiki/concepts/ros2-basics.md)
- 合集索引（历史）：[dds_omg_rtos_edge_ota_safety_primary_refs](dds_omg_rtos_edge_ota_safety_primary_refs.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [fast-dds-docs.md](fast-dds-docs.md) / [repos/fast-dds.md](../repos/fast-dds.md) | eProsima 对 DDS/RTPS 的 C++ 实现 |
| [cyclonedds-io.md](cyclonedds-io.md) / [repos/cyclonedds.md](../repos/cyclonedds.md) | Eclipse 对 DDS/RTPS 的实现 |
| [ros2.md](../repos/ros2.md) | `ros2.repos` 钉定上述 vendor |
