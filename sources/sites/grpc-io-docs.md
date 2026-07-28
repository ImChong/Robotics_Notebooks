# gRPC 官方文档（grpc.io）

> 来源归档

- **标题：** gRPC Documentation — Introduction & Core Concepts
- **类型：** site（官方文档）
- **来源：** gRPC 项目（Google 发起；CNCF incubating）
- **链接：**
  - 首页：https://grpc.io/
  - 文档入口：https://grpc.io/docs/
  - Introduction：https://grpc.io/docs/what-is-grpc/introduction/
  - Core concepts：https://grpc.io/docs/what-is-grpc/core-concepts/
- **代码仓：** https://github.com/grpc/grpc（归档：[repos/grpc.md](../repos/grpc.md)）
- **入库日期：** 2026-07-28
- **一句话说明：** 现代主流 **RPC 框架**的官方概念定义：Protobuf IDL、四种 RPC 形态（unary / 单向流 / 双向流）、同步与异步、deadline、metadata、channel；线协议基于 HTTP/2。
- **沉淀到 wiki：** 是 → [`wiki/entities/grpc.md`](../../wiki/entities/grpc.md)、[`wiki/concepts/remote-procedure-call.md`](../../wiki/concepts/remote-procedure-call.md)

## 为什么值得保留

- 机器人 / 具身栈常见用途：模型服务、边云 API、仿真–工具桥、部分遥操作后端——需要 **一手** 而非博客转述。
- 相对经典 [Birrell & Nelson](../papers/birrell_nelson_implementing_rpc_tocs_1984.md) / [ONC RFC 5531](rfc-5531-onc-rpc.md)，本文档给出 **当代默认工程形态**（Protobuf + HTTP/2 + 多语言 stub）。
- 明确四种方法类型与生命周期，便于与 ROS 2 **Service（近似 unary）** / **Action（长时反馈）** 对照。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 文档站 | **公开可读**（grpc.io；文档源在 grpc.io 站点仓） |
| 运行时 | **已开源** — 见 [repos/grpc.md](../repos/grpc.md)（Apache-2.0） |

## 核心摘录

### 定位（Introduction）

- 客户端可像调用本地对象一样调用另一台机器上服务器的方法。
- 以 **service 定义**为核心：声明可远程调用的方法、参数与返回类型。
- 默认用 **Protocol Buffers** 同时作为 IDL 与消息交换格式（也可换其它格式）。
- 跨语言：例如 Java 服务端 + Go/Python/Ruby 客户端。

### 四种 RPC 方法（Core Concepts）

| 类型 | 语义 |
|------|------|
| **Unary** | 单请求 → 单响应（最接近普通函数调用） |
| **Server streaming** | 单请求 → 响应消息流；单次调用内保序 |
| **Client streaming** | 请求消息流 → 单响应 |
| **Bidirectional streaming** | 双向独立读写流；各自方向保序 |

### 生命周期与其它原语

- **Stub / Client**：客户端本地对象，封装序列化、发送与返回。
- **同步 vs 异步**：同步最接近「过程调用」抽象；网络本质异步，多数语言提供两种 API。
- **Deadline / Timeout**：超时 → `DEADLINE_EXCEEDED`；服务端可查询剩余时间。
- **终止与取消**：两端对成功的判定可不一致；任一方可 cancel（取消不回滚已做副作用）。
- **Metadata**：键值对（鉴权等）；`grpc-` 前缀保留。
- **Channel**：到指定 host:port 的连接抽象；可配压缩等 channel args。

### 协议层（与仓内 CONCEPTS 对齐）

- 抽象 gRPC 调用 = 客户端发起的双向消息流（Call Header → metadata → payloads；对端 Status / trailing metadata）。
- 具体嵌入 **HTTP/2**：流映射、HPACK、长度前缀帧、`END_STREAM` 结束客户端写入。

## 对 wiki 的映射

- 实体：[grpc](../../wiki/entities/grpc.md)
- 概念：[remote-procedure-call](../../wiki/concepts/remote-procedure-call.md)
- 仓：[repos/grpc.md](../repos/grpc.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ROS 2 Services](ros2-official-documentation.md) | 语义上常被描述为请求/响应 RPC；实现走 DDS/RMW，不是 gRPC |
| [网络协议栈](../../wiki/concepts/network-protocol-stack.md) | gRPC 默认 HTTP/2 over TCP；不适合 1 kHz 力矩环 |
| [FreeCAD MCP](../../wiki/entities/freecad-mcp.md) | Addon「RPC」是本地过程暴露，概念同族、实现不同 |

## 推荐继续阅读

- Introduction：<https://grpc.io/docs/what-is-grpc/introduction/>
- Core concepts：<https://grpc.io/docs/what-is-grpc/core-concepts/>
- 仓：<https://github.com/grpc/grpc>
