# MQTT（OASIS）规范一手资料索引

> 来源归档（ingest）

- **标题：** OASIS MQTT 3.1.1 / 5.0 标准与 mqtt.org 官方入口
- **类型：** standard（OASIS Standard）
- **入库日期：** 2026-09-17
- **一句话说明：** MQTT 发布/订阅消息传输协议的权威定义：Client–Server 模型、Topic、三级 QoS、Session/Will、MQTT 5.0 扩展属性；作为 `wiki/concepts/mqtt-protocol.md` 的原始依据。
- **沉淀到 wiki：** 是 → [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)

## 为什么值得保留

- 机器人/嵌入式栈里 **MQTT** 常见于 **ESP32 遥测、云端状态桥、IoT 网关、PlotJuggler 实时流**——需与 **ROS 2/DDS、CAN、LCM** 分层，不能凭博客混谈「轻量 pub/sub」。
- OASIS 规范是 **协议语义与 QoS 保证** 的最终依据；mqtt.org 是 TC 维护的公开入门与生态索引。

## 核心摘录

### 1) OASIS MQTT Version 5.0（2019-03-07，现行主版本）

- **来源：** [MQTT Version 5.0 OASIS Standard](https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html)（PDF/HTML/DOCX 同址）
- **TC：** OASIS Message Queuing Telemetry Transport (MQTT) TC
- **Abstract 要点：**
  - **Client–Server publish/subscribe** 传输协议；轻量、开放、易实现；适合 M2M/IoT 等 **代码 footprint 小、带宽受限** 场景。
  - 运行于 **TCP/IP** 或提供 **有序、无损、双向** 连接的其他网络协议之上。
  - **Payload 与内容无关**（应用自行编码 JSON/CBOR/Protobuf 等）。
  - **三级 QoS：**
    - **QoS 0 — At most once**：尽力交付，可丢；适合高频传感器采样（丢一帧无妨）。
    - **QoS 1 — At least once**：至少一次，**可能重复**；适合需可靠但可去重的指令。
    - **QoS 2 — Exactly once**：恰好一次；适合计费/关键状态（握手开销最大）。
  - **异常断连通知**（Will Message、DISCONNECT reason codes 等）。
- **v5.0 相对 3.1.1 主要扩展（规范 §1.8.2 等）：**
  - Reason Code 与 **Ack 可携带属性**；
  - **Session Expiry**、**Message Expiry**、**Topic Alias**；
  - **User Properties**（键值对扩展头）；
  - **Shared Subscription**（多 Client 负载分担同一 Topic Filter）；
  - **Maximum Packet Size / Receive Maximum** 流控；
  - **Request Response** 模式辅助（通过 Response Topic + Correlation Data）。
- **对 wiki 的映射：** [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)

### 2) OASIS MQTT Version 3.1.1（2014-10-29，广泛部署基线）

- **来源：** [MQTT Version 3.1.1](http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html)
- **要点：**
  - 定义 CONNECT/PUBLISH/SUBSCRIBE 等 **控制报文** 与 **固定头 + 可变头 + 载荷** 结构。
  - **Topic Name**（发布）vs **Topic Filter**（订阅，支持 `+` 单级、`#` 多级通配）。
  - **Clean Session**、**Keep Alive**、**Last Will and Testament**。
  - 大量现网 Broker（含旧版 Mosquitto 默认互操作）仍以 3.1.1 为最低公分母；新部署应优先协商 **5.0** 并回退 3.1.1。
- **对 wiki 的映射：** [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)

### 3) mqtt.org — 标准推广与入门

- **来源：** [mqtt.org](https://mqtt.org/)
- **要点：**
  - 官方称 MQTT 为 **IoT 消息事实标准**；强调 **轻量、双向、可扩展至百万设备、QoS、弱网 Session 恢复、TLS/OAuth 安全**。
  - **Publish/Subscribe 架构图**：Client 只与 **Broker（Server）** 通信，彼此解耦。
  - 行业覆盖：汽车、制造、电信、油气等——机器人侧对应 **fleet 监控、远程运维、边云状态同步**。
- **对 wiki 的映射：** [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)；站点归档 [mqtt-org.md](./mqtt-org.md)

### 4) 术语对照（规范 §1.2，节选）

| 术语 | 含义 |
|------|------|
| **Application Message** | 应用层消息：含 Payload、QoS、Properties、Topic Name |
| **Client** | 发布/订阅/取消订阅的端点 |
| **Server / Broker** | 接受发布、维护订阅、转发 Application Message 的中介 |
| **Session** | Client 与 Server 的有状态交互上下文 |
| **Subscription** | Topic Filter + maximum QoS，绑定 Session |
| **Shared Subscription** | 多 Session 共享同一 Filter，每条消息只投递给一个 Client（v5） |

## 开源核查（2026-09-17）

| 项 | 状态 |
|----|------|
| OASIS 规范文本 | **公开可读**（Non-Assertion IPR；非软件开源许可） |
| 参考 Broker 实现 | **已开源** — 见 [repos/mosquitto.md](../repos/mosquitto.md)（Eclipse Public License 2.0 / EDL） |

> 标准本身不是代码仓；「开源状态」指规范可获取性与常见参考实现。

## 推荐继续阅读（外部）

- OASIS [MQTT TC 主页](https://www.oasis-open.org/committees/mqtt/)
- MQTT 5.0 [新特性概览（非规范）](https://www.oasis-open.org/committees/download.php/66091/MQTT%20v5%20Features%20Overview.pdf)
- [MQTT and the NIST Cybersecurity Framework](http://docs.oasis-open.org/mqtt/mqtt-nist-cybersecurity/v1.0/mqtt-nist-cybersecurity-v1.0.html)

## 当前提炼状态

- [x] 摘要与 wiki 映射
