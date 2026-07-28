# Fast DDS 官方文档（fast-dds.docs.eprosima.com）

> 来源归档

- **标题：** eProsima Fast DDS Documentation
- **类型：** site（官方文档）
- **来源：** eProsima
- **链接：** https://fast-dds.docs.eprosima.com/
- **代码：** https://github.com/eProsima/Fast-DDS（已开源，见 [repos/fast-dds.md](../repos/fast-dds.md)）
- **入库日期：** 2026-07-28
- **一句话说明：** Fast DDS（原 Fast RTPS）的权威文档：DDS/RTPS 双层 API、传输（UDP/TCP/SHM）、Discovery Server、安全、XML QoS、Fast DDS-Gen。
- **沉淀到 wiki：** 是 → [`wiki/entities/fast-dds.md`](../../wiki/entities/fast-dds.md)、[`wiki/concepts/dds-communication.md`](../../wiki/concepts/dds-communication.md)

## 为什么值得保留

- ROS 2 多数 LTS 默认 RMW 实现之一；调 `rmw_fastrtps_cpp` / XML profiles 必须以本站为准。
- 明确区分 **开源 Fast DDS（Apache-2.0）** 与商业 **Fast DDS Pro**（TSN、低带宽、IP Mobility 等）——避免把 Pro 功能当成社区版可用。
- 提供 Installation / User manual / Fast DDS-Gen / CLI / Docker / Release notes 完整导航。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 文档站 → GitHub | ✅ 指向 [eProsima/Fast-DDS](https://github.com/eProsima/Fast-DDS) |
| 社区版开放度 | **已开源**（Apache-2.0） |
| 商业扩展 | Fast DDS Pro（非开源；文档对照表标明 TSN/RPC/IP Mobility 等仅 Pro） |

## 官方自述定位（摘录）

- C++ 实现 OMG **DDS** 规范；提供 **DCPS API** + 底层 **RTPS** 协议实现。
- 被 ROS 2 选为 **每个 LTS 与多数非 LTS** 支持的默认中间件之一。
- 组成：DDS API、**Fast DDS-Gen**（IDL→代码）、RTPS 线协议实现。

## 关键能力（社区版文档列举）

| 能力 | 说明 |
|------|------|
| 双 API 层 | 高层 DDS + 底层 RTPS |
| Discovery | 动态发现；可配 **Discovery Server** / Client-Server |
| 可靠性 | Best Effort / Reliable（可在 UDP 上做可靠语义；亦可走 TCP） |
| 传输 | UDPv4/v6、TCPv4/v6、**SHM**（共享内存） |
| 安全 | 可插拔：认证、访问控制、加密 |
| 配置 | 代码或 **XML profiles**；Flow controllers |
| 序列化 | Fast CDR（CDR，对齐 RTPS） |

## 关键文档入口

| 资源 | URL |
|------|-----|
| 文档首页 | https://fast-dds.docs.eprosima.com/ |
| Linux 二进制安装 | https://fast-dds.docs.eprosima.com/en/latest/installation/binaries/binaries_linux.html |
| Getting Started | https://fast-dds.docs.eprosima.com/en/latest/fastdds/getting_started/getting_started.html |
| Fast DDS-Gen | https://fast-dds.docs.eprosima.com/en/latest/fastddsgen/introduction/introduction.html |
| Fast DDS CLI | https://fast-dds.docs.eprosima.com/en/latest/fastddscli/cli/cli.html |
| Docker | https://fast-dds.docs.eprosima.com/en/latest/docker/docker.html |
| Release notes | https://fast-dds.docs.eprosima.com/en/latest/notes/notes.html |

## 对 wiki 的映射

- 实体：[fast-dds](../../wiki/entities/fast-dds.md)
- 概念：[dds-communication](../../wiki/concepts/dds-communication.md)
- ROS 2：[ros2-basics](../../wiki/concepts/ros2-basics.md)
- 规范：[omg-dds-spec](omg-dds-spec.md)
- 代码仓：[repos/fast-dds.md](../repos/fast-dds.md)
