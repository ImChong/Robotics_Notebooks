# protocolbuffers/protobuf

> 来源归档

- **标题：** Protocol Buffers — Google's data interchange format
- **类型：** repo
- **来源：** Google LLC（2008 起开源）
- **链接：** https://github.com/protocolbuffers/protobuf
- **Homepage：** https://protobuf.dev/（文档归档：[sites/protobuf-dev-docs.md](../sites/protobuf-dev-docs.md)）
- **Stars：** ~72k（2026-09）
- **默认分支：** `main`
- **许可证：** BSD-3-Clause（Copyright 2008 Google LLC）
- **入库日期：** 2026-09-17
- **一句话说明：** 跨语言 **结构化数据序列化** 运行时 + **`protoc` 编译器**：`.proto` IDL → 各语言类型与编解码；gRPC 默认载荷与契约格式。
- **沉淀到 wiki：** 是 → [`wiki/entities/protocol-buffers.md`](../../wiki/entities/protocol-buffers.md)

## 开源状态（2026-09-17）

**已开源**：本仓含 C++ 核心、`protoc` 编译器、Java/Python/C#/Ruby/Objective-C/PHP 等运行时；Go / Dart / JavaScript 等语言在独立 GitHub 仓维护，由 `protoc` 插件生成代码。Release 页提供各平台 `protoc-$VERSION-$PLATFORM.zip` 预编译包。

## README 定位（摘要）

- **语言中立、平台中立** 的可扩展结构化数据序列化机制；比 JSON 更小更快，并生成原生语言绑定。
- 组合四部分：**`.proto` 定义语言**、**`protoc` 生成代码**、**语言运行时库**、**二进制 wire format**。
- 典型用途：**RPC 契约与载荷**（与 [gRPC](grpc.md) 搭配）、**磁盘/网络持久化**、跨项目共享 `message` 类型（如 `timestamp.proto`）。
- 向后兼容：新增字段旧代码可读（忽略未知字段）；删除字段应 **reserve** 字段号。

## 仓库内容结构（导读）

| 路径 | 作用 |
|------|------|
| `src/` | C++ 运行时 + `protoc` 编译器 |
| `java/`、`python/`、`csharp/` 等 | 各语言运行时 |
| `examples/` | 入门示例 |
| `docs/` | 文档源（发布到 protobuf.dev） |

## 编译器与运行时安装（官方 README）

| 需求 | 入口 |
|------|------|
| 仅 `protoc` | [GitHub Releases](https://github.com/protocolbuffers/protobuf/releases) 下载 `protoc-$VERSION-$PLATFORM.zip` |
| C++ 从源码 | `src/README.md` |
| Java / Python / … | 各语言子目录 + 语言包管理器（如 `pip install protobuf`） |
| Go | [protocolbuffers/protobuf-go](https://github.com/protocolbuffers/protobuf-go) |
| Dart | [dart-lang/protobuf](https://github.com/dart-lang/protobuf) |
| JavaScript | [protocolbuffers/protobuf-javascript](https://github.com/protocolbuffers/protobuf-javascript) |

版本支持策略见 <https://protobuf.dev/version-support/>。

## 对 wiki 的映射

- [Protocol Buffers 实体](../../wiki/entities/protocol-buffers.md)
- [gRPC 实体](../../wiki/entities/grpc.md)（默认 IDL + 载荷）
- [远程过程调用概念](../../wiki/concepts/remote-procedure-call.md)
- 文档站：[protobuf-dev-docs.md](../sites/protobuf-dev-docs.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [grpc/grpc](grpc.md) | gRPC 默认 Protobuf 作 service/message 定义与序列化 |
| [ONNX Runtime](onnxruntime-v1.28.0.md) | ONNX 1.22 捆绑 protobuf 6.33.5 作模型格式依赖 |
| [LCM](../concepts/lcm-basics.md) | 同为 IDL + 强类型序列化，但 UDP 组播 pub/sub，非 Protobuf wire |
| [rhoban_bam](rhoban_bam.md) | eRob 采集示例需 `generate_protobuf.sh` 生成 gRPC stub |

## 推荐继续阅读

- 仓：<https://github.com/protocolbuffers/protobuf>
- 文档：<https://protobuf.dev/>
- Getting started：<https://protobuf.dev/getting-started/>
- 版本支持：<https://protobuf.dev/version-support/>
