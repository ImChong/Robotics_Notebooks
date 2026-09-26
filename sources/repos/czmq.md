# zeromq/czmq

> 来源归档

- **标题：** CZMQ — High-level C binding for ZeroMQ
- **类型：** repo
- **链接：** https://github.com/zeromq/czmq
- **许可证：** MPL-2.0
- **入库日期：** 2026-09-26
- **一句话说明：** 在 libzmq 之上的 **高级 C 封装**：actor、zmsg/zframe、zloop/zpoller 等，降低 C 项目中的内存与 socket 生命周期管理成本。
- **沉淀到 wiki：** 是 → [`wiki/entities/zeromq.md`](../../wiki/entities/zeromq.md)（可选 C 栈入口）

## 开源状态（2026-09-26）

**已开源**：与 libzmq 同社区维护；部分语言绑定（如 czmq 生态）依赖此层。

## 对 wiki 的映射

- 机器人 C/C++ 部署栈若已用 libzmq 原生 API，可不引入 czmq；嵌入式网关或自研中间层可参考 czmq 的消息对象模型。
