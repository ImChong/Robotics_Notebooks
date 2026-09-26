# The ØMQ Guide（zguide.zeromq.org）

> 来源归档

- **标题：** ZeroMQ Guide — Learn ZeroMQ step by step
- **类型：** site（官方教程 / 一手教材）
- **链接：** https://zguide.zeromq.org/
- **纸质书：** O'Reilly *ZeroMQ*（Pieter Hintjens；与在线 Guide 同源）
- **入库日期：** 2026-09-26
- **一句话说明：** 官方 **从入门到进阶** 的 ZeroMQ 使用指南：60+ 图、750+ 多语言示例，覆盖 **socket 模式组合**、可靠性、拓扑与性能调优——工程上选型与写 Client/Server 的默认参考书。
- **沉淀到 wiki：** 是 → [`wiki/concepts/zeromq-messaging.md`](../../wiki/concepts/zeromq-messaging.md)

## 为什么值得保留

- API 行为（如 REQ 的 lockstep、SUB 慢 joiner）在 Guide 中有 **场景化解释**，比仅读 RFC 更易避免经典死锁/丢首包问题。
- 与 [zeromq.org Get started](../sites/zeromq-org-primary-refs.md) 互为入口：站点给绑定列表，Guide 给 **模式与架构**。

## 核心摘录（结构级，非全文转存）

### Guide 覆盖的典型章节主题

- **基础 socket 类型** 与 **组合拓扑**（request–reply、pub–sub、pipeline、exclusive pair）。
- **消息 envelope** 与 **multipart**；ROUTER/DEALER 路由 identity。
- **可靠性模式**：重连、心跳、幂等、中间层 queue device（历史 device 概念；现代多直接用 socket 组合）。
- **并发与线程**：inproc 同进程、跨进程 ipc/tcp；与机器人 **仿真进程 ↔ 策略 GPU 进程** 拆分一致。
- **安全章节**：PLAIN/CURVE 与 trust model（细节线协议见 [zmq-rfc-zmtp-3.md](zmq-rfc-zmtp-3.md)）。

## 对 wiki 的映射

- 概念页「常见误区」：慢 joiner、REQ 死锁、把 ZeroMQ 当持久化队列等，引用 Guide 章节名即可，不复制大段示例代码。
