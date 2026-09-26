# ZeroMQ RFC — ZMTP 3.0（spec:23）

> 来源归档

- **标题：** ZeroMQ Message Transport Protocol (ZMTP) 3.0
- **类型：** site（协议规范 / RFC）
- **链接：** https://rfc.zeromq.org/spec:23/
- **状态：** stable（规范站标注）
- **编辑：** Pieter Hintjens 等
- **入库日期：** 2026-09-26
- **一句话说明：** ZeroMQ 在 TCP 等 **面向连接传输** 上的 **线协议**：帧定界、多 part 标志、版本协商、可插拔安全机制（PLAIN/CURVE 等）与连接元数据——实现互操作性的 **一手** 依据。
- **沉淀到 wiki：** 是 → [`wiki/concepts/zeromq-messaging.md`](../../wiki/concepts/zeromq-messaging.md)

## 为什么值得保留

- Socket API（PUB/SUB、REQ/REP…）的 **语义** 在独立 RFC（spec:28–31 等）中定义；**ZMTP** 定义 **线上如何成帧与安全握手**，排查跨语言/跨版本互通问题需读此层。
- 明确 ZMTP 3.0 相对 2.0 的变更：安全机制、 greeting 中移除硬编码 socket type/identity、增加 commands 与连接 metadata。

## 核心摘录

### 解决的问题

- TCP 是 **无消息边界** 的字节流 → ZMTP 读写 **size + body** 帧。
- 每帧带 **flags**（如 multipart 续帧）。
- **Greeting** 宣布版本并支持 **版本协商**。
- **Security handshake**：从明文到全认证加密；机制可扩展（PLAIN、CURVE、ZAP 等，见 spec:24–27）。
- 连接 metadata（socket type、identity 等）在握手后交换。

### 相关规范（规范站索引）

| RFC | 主题 |
|-----|------|
| spec:24 | ZMTP-PLAIN |
| spec:25 | ZMTP-CURVE |
| spec:26 | CurveZMQ |
| spec:27 | ZAP（认证协议） |
| spec:28 | REQ/REP/DEALER/ROUTER 语义 |
| spec:29 | PUB/SUB/XPUB/XSUB |
| spec:30 | PUSH/PULL pipeline |
| spec:31 | exclusive PAIR |
| spec:37 | ZMTP 3.1 |

## 对 wiki 的映射

- 概念页「传输与安全」小节：ZMTP 帧、CURVE 在边云/跨机策略推理中的含义（加密开销 vs 明文 IPC）。
