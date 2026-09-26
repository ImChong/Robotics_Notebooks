# ZeroMQ 官方网站（zeromq.org）

> 来源归档

- **标题：** ZeroMQ — Get started & community docs
- **类型：** site（官方站点）
- **链接：**
  - 首页 / Get started：https://zeromq.org/get-started/
  - 语言绑定索引：https://zeromq.org/languages/
  - 下载：https://zeromq.org/download/
- **代码：** [libzmq](../repos/libzmq.md)（C/C++ 核心）；[czmq](../repos/czmq.md)（高级 C API）
- **入库日期：** 2026-09-26
- **一句话说明：** **无中心 Broker** 的高性能异步消息库：在 TCP / IPC / inproc / multicast / WebSocket 等传输上实现 pub/sub、req/rep、pipeline 等 **socket 模式**；机器人栈中常见于 **策略 Server ↔ 仿真/真机 Client** 的轻量 RPC/流式载荷通道。
- **沉淀到 wiki：** 是 → [`wiki/entities/zeromq.md`](../../wiki/entities/zeromq.md)、[`wiki/concepts/zeromq-messaging.md`](../../wiki/concepts/zeromq-messaging.md)

## 为什么值得保留

- 官方对「Zero」哲学、**brokerless** 与 **scale by composition** 的定义是一手表述，避免二手博客混淆 ZeroMQ 与 Kafka/RabbitMQ。
- 绑定矩阵（PyZMQ、JeroMQ、NetMQ 等）与传输/安全能力列表以站点为准，便于核对某语言是否支持 CURVE / GSSAPI。
- 与仓内 [MQTT 概念页](../../wiki/concepts/mqtt-protocol.md)（Client–Broker）、[gRPC](../../wiki/entities/grpc.md)（IDL + HTTP/2 RPC）形成 **中间件三角**对照。

## 开源核查（2026-09-26）

| 项 | 状态 |
|----|------|
| 站点文档 | **公开可读**（zeromq.org） |
| 核心实现 | **已开源** — [zeromq/libzmq](https://github.com/zeromq/libzmq)（MPL-2.0） |
| 规范 | **公开** — [ZeroMQ RFCs](https://rfc.zeromq.org/)（ZMTP 线协议与 socket 语义）；归档 [zmq-rfc-zmtp-3.md](zmq-rfc-zmtp-3.md) |

## 核心摘录（Get started）

### 定位

- ZeroMQ（ØMQ / 0MQ / ZMQ）是面向 **分布式或并发应用** 的高性能 **异步消息库**。
- 提供消息队列能力，但 **Unlike message-oriented middleware**，系统可在 **无专用 message broker** 下运行。
- 在多种传输上支持 **常见消息模式**（pub/sub、request/reply、client/server 等），使 **进程间** 通信复杂度接近 **线程间** 通信。

### 「Zero」含义（官方哲学）

- **Zero broker**、低延迟、零成本（免费）、零运维；更广义指 **极简主义**：通过 **减复杂度** 而非堆功能来增强能力。

### 官方学习路径

- **The Guide**（[zguide.zeromq.org](https://zguide.zeromq.org/)）：基础到高级，含多语言示例；归档 [zguide-zeromq.md](zguide-zeromq.md)。
- **Libzmq**：多数语言绑定的底层 C/C++ 库；贡献与内部机制从 libzmq 读起。

## 对 wiki 的映射

- 实体页：项目、许可证、机器人侧典型用法（GR00T Policy Server、RLDX server、PlotJuggler 流等已入库案例）。
- 概念页：socket 类型语义、与 MQTT/ROS 2/DDS 分层对照、选型边界（非 1 kHz 硬实时总线）。
