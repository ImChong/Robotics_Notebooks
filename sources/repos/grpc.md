# grpc/grpc

> 来源归档

- **标题：** gRPC — An RPC library and framework
- **类型：** repo
- **来源：** gRPC Authors（Google 发起；CNCF incubating）
- **链接：** https://github.com/grpc/grpc
- **Homepage：** https://grpc.io/（文档归档：[sites/grpc-io-docs.md](../sites/grpc-io-docs.md)）
- **Stars：** ~45.2k（2026-07）
- **默认分支：** `master`
- **最新发行：** v1.83.0（2026-07-22）
- **许可证：** Apache-2.0
- **入库日期：** 2026-07-28
- **一句话说明：** 多语言 **gRPC** 运行时与共享 C++ 核心：在 HTTP/2 上实现高性能 RPC；默认 Protobuf IDL；提供 examples 与各语言包入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/grpc.md`](../../wiki/entities/grpc.md)

## 开源状态（2026-07-28）

**已开源**：本仓含多语言绑定（基于 `src/core` C++ 核心）、examples、协议文档（如 `doc/PROTOCOL-HTTP2.md`、`CONCEPTS.md`）。各语言另有独立发行仓/包（`grpc-go`、`grpc-java`、`grpc-dotnet` 等）——生产依赖通常走语言包管理器，而非从本仓整树编译。

## README / CONCEPTS 定位（摘要）

- 现代开源高性能 RPC；可在任意环境运行；支持跨语言透明调用。
- 从 **语言无关的 service 描述**生成客户端/服务端接口；服务端实现接口，客户端通过 stub 远程调用。
- 默认 [Protocol Buffers](https://github.com/protocolbuffers/protobuf) 作 IDL 与载荷；可替换。
- 支持 **同步 / 异步** API；支持 **streaming**（含双向流，单次调用内保序）。
- 协议：抽象要求见仓内文档；具体嵌入 **HTTP/2**（流、HPACK、flow control）。

## 各语言起步（官方 README 表）

| 语言 | 入口 |
|------|------|
| C++ | 本仓 `src/cpp` |
| C# / .NET | `grpc/grpc-dotnet`（NuGet） |
| Dart | `grpc/grpc-dart` |
| Go | `google.golang.org/grpc` |
| Java / Kotlin | Maven Central（`grpc-java` / `grpc-kotlin`） |
| Node | `@grpc/grpc-js` |
| Python | `pip install grpcio`（`src/python/grpcio`） |
| 其它 | PHP pecl、Ruby gem、Objective-C pod、grpc-web 等 |

示例：本仓 `examples/`；教程：https://grpc.io/docs/

## 仓库内容结构（导读）

| 路径 | 作用 |
|------|------|
| `src/core` | 共享 C++ 核心 |
| `src/cpp` 等 | 语言运行时 |
| `examples/` | 跨语言示例 |
| `doc/PROTOCOL-HTTP2.md` | HTTP/2 映射细节 |
| `CONCEPTS.md` | 概念总览（与文档站对齐） |
| `BUILDING.md` / `CONTRIBUTING.md` | 从源码构建与贡献 |

## 对 wiki 的映射

- [gRPC 实体](../../wiki/entities/grpc.md)
- [远程过程调用概念](../../wiki/concepts/remote-procedure-call.md)
- 文档站：[grpc-io-docs.md](../sites/grpc-io-docs.md)
- 概念源头：[birrell_nelson_implementing_rpc_tocs_1984.md](../papers/birrell_nelson_implementing_rpc_tocs_1984.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [ONC RFC 5531](../sites/rfc-5531-onc-rpc.md) | 另一历史 RPC 线协议；与 gRPC 不互通 |
| [ROS 2 / RMW](rmw.md) | ROS Service 是 RPC *风格* 原语，默认实现不是本仓 |
| [robot-io-rio](robot-io-rio.md) | 中间件可切换列表含 ZeroRPC 等；同属 RPC 家族工程选项 |
| [rhoban_bam](rhoban_bam.md) | 采集路径示例：gRPC 经 Etherban |

## 推荐继续阅读

- 仓：<https://github.com/grpc/grpc>
- 文档：<https://grpc.io/docs/>
- 性能看板：README 链出的 Grafana dashboard
