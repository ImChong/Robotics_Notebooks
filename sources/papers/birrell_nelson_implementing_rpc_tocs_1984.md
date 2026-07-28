# Implementing Remote Procedure Calls（Birrell & Nelson，TOCS 1984）

> 来源归档

- **标题：** Implementing Remote Procedure Calls
- **类型：** paper（经典一手）
- **作者：** Andrew D. Birrell, Bruce Jay Nelson
- **机构：** Xerox Palo Alto Research Center（Xerox PARC）
- **出处：** ACM Transactions on Computer Systems, Vol. 2, No. 1, February 1984, pp. 39–59
- **DOI / 检索：** ACM 0734-2071/84/0200-0039；常见镜像 [birrell.org PDF](http://birrell.org/andrew/papers/ImplementingRPC.pdf)
- **入库日期：** 2026-07-28
- **一句话说明：** 把 **本地过程调用语义扩展到网络**：挂起调用方、参数跨机传递、被调用方执行、结果返回后恢复——奠定现代 RPC / stub / binding / 故障语义讨论的经典工程报告。
- **沉淀到 wiki：** 是 → [`wiki/concepts/remote-procedure-call.md`](../../wiki/concepts/remote-procedure-call.md)

## 为什么值得保留

- IETF [RFC 5531](../sites/rfc-5531-onc-rpc.md) 明确推荐本文作为 RPC **概念背景**；后续 ONC RPC、gRPC、ROS Service 等均继承「像本地调用一样远程调用」的叙事。
- 系统列出设计者必须回答的问题：**失败语义、含地址参数、语言集成、绑定、传输协议、安全**——比二手博客更适合做概念页骨架。
- 本库通信链（DDS / RMW / ROS 2 Services）需要把 **请求–响应 RPC** 与 **pub/sub** 对照；本文是 RPC 侧的源头一手。

## 开源核查（2026-07-28）

| 项 | 状态 |
|----|------|
| 论文 PDF | **公开可读**（作者站 / 多校镜像；版权归 ACM，非软件许可） |
| 历史实现 | Cedar / Dorado 环境实验包；**非**现代可复现机器人依赖 |

> 本文是概念与系统设计一手，不是可运行机器人中间件仓。

## 核心摘录

### RPC 基本模型（§1.1）

- 过程调用本是单机内传递控制与数据的熟知机制；RPC 将其扩展到通信网络。
- 调用时：**调用环境挂起** → 参数传到被调用环境（callee）→ 执行过程 → **结果回传** → 调用环境像普通返回一样继续。
- 宣称吸引力：清晰语义、可做得很快、通用（单机程序大量靠过程通信）。

### 设计者必须面对的议题（§1.1）

| 议题 | 含义 |
|------|------|
| 失败语义 | 机器/网络故障时一次调用意味着什么（至少一次 / 至多一次等） |
| 含地址参数 | 无共享地址空间时指针/引用如何处理 |
| 语言集成 | 如何嵌入现有或未来编程系统 |
| Binding | 调用方如何确定 callee 的位置与身份 |
| 传输协议 | caller↔callee 的数据与控制传递 |
| 完整性与安全 | 开放网络上的完整性与保密（可选） |

### 结构与工程产物（论文主张）

- 描述 **整体结构**、**客户端绑定**、**传输层协议**与性能测量。
- **Stub**：负责参数与结果解释（制造 stub 模块的细节文中预告后续专文）。
- Cedar 环境 + Xerox 研究互联网络上的实现经验；强调相对「纸面设计」，全规模实现更少见。

## 对 wiki 的映射

- 主概念：[remote-procedure-call](../../wiki/concepts/remote-procedure-call.md)
- 现代实现实体：[grpc](../../wiki/entities/grpc.md)
- 线协议标准层：[rfc-5531-onc-rpc](../sites/rfc-5531-onc-rpc.md)
- 机器人对照：[ros2-basics](../../wiki/concepts/ros2-basics.md)（Services ≈ 请求/响应 RPC）

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [RFC 5531 ONC RPC](../sites/rfc-5531-onc-rpc.md) | 标准线协议；引言直接引用本文为背景 |
| [gRPC 文档 / 仓](../sites/grpc-io-docs.md) | 现代主流 RPC 框架；语义仍是「像本地对象一样调用远程方法」 |
| Regularized Predictive Control 的 `paper-*-rpc*` | **同缩写异义**；腿足控制线，勿与本页 Remote Procedure Call 混淆 |

## 推荐继续阅读

- PDF：<http://birrell.org/andrew/papers/ImplementingRPC.pdf>
- RFC 5531：<https://www.rfc-editor.org/rfc/rfc5531>
- gRPC Concepts：<https://grpc.io/docs/what-is-grpc/core-concepts/>
