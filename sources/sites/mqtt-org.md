# mqtt.org 官方站点

> 来源归档

- **标题：** MQTT — The Standard for IoT Messaging（mqtt.org）
- **类型：** site（标准组织维护的公开入门站）
- **链接：** https://mqtt.org/
- **维护方：** OASIS MQTT TC 生态推广（非替代规范正文）
- **入库日期：** 2026-09-17
- **一句话说明：** MQTT 协议的官方公开入口：定位 IoT 轻量 pub/sub、QoS 三级、TLS/OAuth 安全与行业采用；链向 OASIS 规范与入门资源。
- **沉淀到 wiki：** 是 → [`wiki/concepts/mqtt-protocol.md`](../../wiki/concepts/mqtt-protocol.md)

## 为什么值得保留

- 区分 **mqtt.org（入门与生态）** 与 **docs.oasis-open.org（规范正文）**——调试互操作问题应以后者为准。
- 站点强调的设计目标（小 footprint、低带宽、弱网、百万连接）直接对应机器人 **遥测/运维/边云** 层，而非 1 kHz 关节环。

## 核心摘录

### 官方定位（首页，2026-09 复核）

| 特性 | 说明 |
|------|------|
| **Lightweight** | Client 极小，适合 MCU；报文头紧凑 |
| **Bi-directional** | 设备 ↔ 云双向；便于广播与命令下发 |
| **Scale** | 可扩展至百万级连接 |
| **Reliable delivery** | QoS 0/1/2 三级语义 |
| **Unreliable networks** | 持久 Session 缩短重连恢复时间 |
| **Security** | TLS 加密；OAuth 等现代认证（v5 生态） |

### 架构（站点图示语义）

- **Publish/Subscribe**：Producer 不直连 Consumer；**Broker** 按 Topic 路由。
- 与 **ROS 2 DDS**（无中心 Broker、peer 发现）和 **LCM**（UDP 组播）形成对照——见 [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md) 工程分层。

## 开源核查（2026-09-17）

| 项 | 状态 |
|----|------|
| 站点内容 | **公开可读** |
| 协议实现 | 见 [mosquitto.md](../repos/mosquitto.md) 等独立开源项目 |

## 对 wiki 的映射

- [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)
- [hub-communication](../../wiki/overview/hub-communication.md)（IoT/遥测层）

## 当前提炼状态

- [x] 摘要与 wiki 映射
