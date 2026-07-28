# RFC 5531：ONC RPC Protocol Version 2（一手标准）

> 来源归档

- **标题：** RPC: Remote Procedure Call Protocol Specification Version 2
- **类型：** site（IETF Standards Track RFC）
- **来源：** IETF / Network Working Group（作者 R. Thurlow, Sun Microsystems）
- **RFC：** [RFC 5531](https://www.rfc-editor.org/rfc/rfc5531)（2009-05；Obsoletes RFC 1831）
- **HTML：** https://www.rfc-editor.org/rfc/rfc5531.html
- **Info：** https://www.rfc-editor.org/info/rfc5531/
- **入库日期：** 2026-07-28
- **一句话说明：** **Open Network Computing (ONC) RPC v2** 线上消息协议的权威定义（用 XDR 描述）；区分「RPC 概念模型」与「可互操作的线格式」，并明确推荐 Birrell & Nelson 作为概念背景。
- **沉淀到 wiki：** 是 → [`wiki/concepts/remote-procedure-call.md`](../../wiki/concepts/remote-procedure-call.md)

## 为什么值得保留

- 与 [Birrell & Nelson 1984](../papers/birrell_nelson_implementing_rpc_tocs_1984.md) 形成 **概念层 + 协议层** 一手对：前者讲设计议题，本 RFC 讲 ONC 部署中的报文与程序/过程编号。
- 机器人工程中 NFS、部分遗留工具链仍可能碰到 ONC RPC；更重要的是理解：**「叫 RPC」不等于只有一种线协议**（ONC ≠ gRPC ≠ ROS Service）。
- 明确 **binding 独立于 RPC 协议本身**（§6）：绑定/汇合可另选机制——与 ROS Service 发现、gRPC channel 选型对照有用。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 规范文本 | **公开可读**（RFC Editor；IETF Trust 版权，非软件开源许可） |
| 参考实现 | 不在本页；现代机器人栈更常接触 [gRPC](../repos/grpc.md) |

## 核心摘录

### 定位（Abstract / §1）

- 描述当前部署并被接受的 **ONC RPC version 2** 消息协议。
- **不试图**论证「为何使用 RPC」或描述一般用法；概念背景指向 Birrell & Nelson [*Implementing Remote Procedure Calls*](../papers/birrell_nelson_implementing_rpc_tocs_1984.md)。
- 消息协议用 **XDR**（RFC 4506）描述。

### RPC 模型要点（§4–6 摘要）

| 概念 | 说明 |
|------|------|
| Caller / Server | 调用方发请求；服务方执行过程并（通常）回传结果 |
| 传输无关 | 可跑在多种传输之上；语义受传输可靠性影响 |
| At-most-once vs 其它 | 协议与传输组合决定重复/丢失语义；应用需理解 |
| Binding 独立 | 程序号/版本/过程号标识调用目标；如何发现端点可独立实现 |
| Authentication | 多种 auth flavor；Null 认证等（§10） |

### 协议要素（§8–9）

- **Program / Procedure / Version** 编号空间（IANA 分配策略见 §13）。
- 两类消息：CALL 与 REPLY；含 xid 等用于匹配。
- 可支持 batching、broadcast RPC 等扩展用法（§8.4）。

## 对 wiki 的映射

- 主概念：[remote-procedure-call](../../wiki/concepts/remote-procedure-call.md)
- 概念源头论文：[birrell_nelson_implementing_rpc_tocs_1984](../papers/birrell_nelson_implementing_rpc_tocs_1984.md)
- 现代框架：[grpc](../../wiki/entities/grpc.md) / [grpc-io-docs](grpc-io-docs.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [gRPC](../repos/grpc.md) | 另一套 RPC 栈（HTTP/2 + Protobuf）；**不是** ONC 超集 |
| [网络协议栈](../../wiki/concepts/network-protocol-stack.md) | ONC 可承载于 TCP/UDP；与控制环 UDP 选型对照 |
| [DDS](omg-dds-spec.md) | 数据中心化 pub/sub，不是过程调用模型 |

## 推荐继续阅读

- RFC 5531：<https://www.rfc-editor.org/rfc/rfc5531>
- XDR：RFC 4506
- gRPC 核心概念：<https://grpc.io/docs/what-is-grpc/core-concepts/>
