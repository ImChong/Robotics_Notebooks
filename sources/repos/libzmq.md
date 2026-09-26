# zeromq/libzmq

> 来源归档

- **标题：** libzmq — ZeroMQ core library in C++
- **类型：** repo
- **链接：** https://github.com/zeromq/libzmq
- **Homepage：** https://zeromq.org/（[sites/zeromq-org-primary-refs.md](../sites/zeromq-org-primary-refs.md)）
- **许可证：** MPL-2.0
- **入库日期：** 2026-09-26
- **一句话说明：** ZeroMQ **底层核心**：暴露 **C API**（C++ 实现）；各语言绑定多基于此；实现 ZMTP 与 socket 引擎。
- **沉淀到 wiki：** 是 → [`wiki/entities/zeromq.md`](../../wiki/entities/zeromq.md)

## 开源状态（2026-09-26）

**已开源**：完整源码、CMake 构建、tests 与 `include/zmq.h` 公共 API。生产环境通常通过 **系统包**（`libzmq3-dev`）或 **语言绑定**（PyZMQ 等）依赖本库，而非直接 fork 核心。

## README 定位（摘要）

- 高性能异步消息库，**brokerless**。
- 支持 tcp、ipc、inproc、pgm、norm、websocket 等传输（具体可用性随版本与构建选项）。
- 安全：PLAIN、CURVE（libsodium）、GSSAPI 等机制（与 ZMTP RFC 对齐）。

## 与机器人仓内用例

| 场景 | 典型模式 | 站内页 |
|------|----------|--------|
| GR00T Policy Server ↔ Arena | TCP req/rep 或自定义帧 | [isaac-gr00t](../../wiki/entities/isaac-gr00t.md)、[nvidia-gr00t-e2e-g1-workflow](../../wiki/entities/nvidia-gr00t-e2e-g1-workflow.md) |
| RLDX 远程推理 | `run_rldx_server.py` | [rldx-1](../../wiki/entities/rldx-1.md) |
| 遥操作 / 相机流 | pub/sub 或 push/pull | [xr-teleoperate](../../wiki/entities/xr-teleoperate.md)、[paper-pi-r2](../../wiki/entities/paper-pi-r2.md) |
| 调试可视化 | PlotJuggler ZMQ 插件 | [plotjuggler](../../wiki/entities/plotjuggler.md) |

## 对 wiki 的映射

- 实体页「工程实践」：优先 **语言绑定 + 官方 Guide 模式**；仅在做性能/互通调试时深入 libzmq。
