# Protocol Buffers 官方文档（protobuf.dev）

> 来源归档

- **标题：** Protocol Buffers Documentation — Overview, Proto3, Encoding
- **类型：** site（官方文档）
- **来源：** Google / Protocol Buffers 开源项目
- **链接：**
  - 首页：https://protobuf.dev/
  - Overview：https://protobuf.dev/overview/
  - Proto3 语言指南：https://protobuf.dev/programming-guides/proto3/
  - Wire 编码：https://protobuf.dev/programming-guides/encoding/
  - Getting started：https://protobuf.dev/getting-started/
- **代码仓：** https://github.com/protocolbuffers/protobuf（归档：[repos/protobuf.md](../repos/protobuf.md)）
- **入库日期：** 2026-09-17
- **一句话说明：** **结构化数据序列化** 的一手定义：`.proto` 语法、跨语言代码生成、varint/TLV wire format、字段演进与兼容规则；gRPC 默认契约与载荷格式。
- **沉淀到 wiki：** 是 → [`wiki/entities/protocol-buffers.md`](../../wiki/entities/protocol-buffers.md)

## 为什么值得保留

- 机器人栈中 **gRPC 服务 API、边云推理接口、部分驱动/采集桥** 默认 Protobuf；需要 **一手** 理解 IDL、兼容性与 wire 行为，而非博客转述。
- 与 [JSON](https://www.json.org/)、[ONC XDR](../sites/rfc-5531-onc-rpc.md)、[LCM `.lcm` 类型](../concepts/lcm-basics.md) 对照时，Protobuf 的 **字段号 + 未知字段跳过** 是工程演进关键。
- ONNX 等格式底层亦依赖 protobuf；读 [ONNX Runtime](../repos/onnxruntime-v1.28.0.md) 版本说明时需知 protobuf 版本钉扎含义。

## 开源核查（2026-09-17）

| 项 | 状态 |
|----|------|
| 文档站 | **公开可读**（protobuf.dev） |
| 编译器与运行时 | **已开源** — 见 [repos/protobuf.md](../repos/protobuf.md)（BSD-3-Clause） |

## 核心摘录

### 定位（Overview）

- **语言中立、平台中立** 的可扩展结构化数据序列化；类似 JSON 但更小更快，并生成原生语言绑定。
- 四要素：**`.proto` 定义**、**`protoc` 生成代码**、**语言运行时**、**二进制序列化格式**。
- 适用：**数 MB 以内** 的 typed structured packets；网络 ephemeral 流量与磁盘长期存储均可。
- **向后兼容**：新增字段旧代码可读；删除字段旧代码见默认值/空 repeated；新代码读旧消息时新字段取默认值。
- **不适合**：单消息需全量进内存且 **> 数 MB**；需无损比较二进制而不解析；科学计算大 float 数组（不如 FITS）；非 OO 语言（Fortran/IDL）；**无法律级「正式标准」** 要求的场景。

### 工作流（Overview 图）

1. 编写 `.proto`（`message` / `enum` / `service` 等）。
2. 构建时 **`protoc`** 生成各语言类（getter/setter、`serialize`/`parse`）。
3. 应用读写 **文件流或网络字节**。

### 语法要点（Overview + Proto3）

- **Cardinality**：singular / repeated；proto3 可选 `optional` 改 **explicit presence**。
- **类型**：标量、`message` 嵌套、`enum`、`oneof`、`map`、extensions（库内选项）。
- **字段号**：一经使用 **不可复用**；删除字段应 **reserve** 编号。
- **Edition 2023**：新语法线（`edition = "2023"`）；与 proto2/proto3 并存，见语言指南。

### Wire 编码（Encoding）

- 消息 = **field number + wire type + payload** 的 **TLV** 序列；解析端靠 `.proto` 还原类型与名称。
- **Varint**：变长无符号整数，小值占字节少（如 `150` → `96 01`）。
- **Wire types**：`VARINT`、`I64`、`LEN`、`SGROUP`、`EGROUP`、`I32`。
- 旧 parser 可 **跳过未知字段** → 演进兼容的基础。
- 同一逻辑数据可有多种合法二进制序列 → **不能直接 memcmp 判等**，须 parse 后比较。

### 生态（Overview）

- 广泛使用：**gRPC**、Google Cloud、Envoy Proxy 等。

## 对 wiki 的映射

- 实体：[protocol-buffers](../../wiki/entities/protocol-buffers.md)
- 关联：[grpc](../../wiki/entities/grpc.md)、[remote-procedure-call](../../wiki/concepts/remote-procedure-call.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [grpc-io-docs](grpc-io-docs.md) | gRPC 默认 Protobuf 同时作 IDL 与 message 格式 |
| [ONNX Runtime v1.28](../repos/onnxruntime-v1.28.0.md) | 捆绑 protobuf 6.33.5 |
| [LCM 基础](../../wiki/concepts/lcm-basics.md) | 另一套 IDL + 序列化 + UDP 组播，非 Protobuf |

## 推荐继续阅读

- Overview：<https://protobuf.dev/overview/>
- Encoding：<https://protobuf.dev/programming-guides/encoding/>
- Proto3 指南：<https://protobuf.dev/programming-guides/proto3/>
- 仓：<https://github.com/protocolbuffers/protobuf>
