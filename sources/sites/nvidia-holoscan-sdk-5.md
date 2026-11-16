# NVIDIA Holoscan SDK 5.0 — 官方技术文档（Official Documentation）

> NVIDIA 官方技术文档一手资料归档；核对日期：2026-11-12。

- **标题：** NVIDIA Holoscan SDK 5.0
- **类型：** site（NVIDIA 官方 SDK 文档）
- **项目：** <https://developer.nvidia.com/holoscan-sdk>
- **文档：** <https://docs.nvidia.com/holoscan/sdk-user-guide/>
- **版本：** 5.0（对应源码 tag `v5.0.0`，2026-10-21 发布）
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sdk>
- **入库日期：** 2026-11-12
- **核对范围：** 通读官方 Overview、Core Concepts、Temporal Contracts、SDK Installation、Relevant Technologies、Slow/Fast Fusion 与 Camera + IMU Control 示例的主体机制、运行方式及限制；并对照 v5.0.0 示例 README 与 C++ 入口。并非声称通读官方文档网站全部章节。
- **通读：** 已完成
- **沉淀到 wiki：** [NVIDIA Holoscan SDK](../../wiki/entities/nvidia-holoscan-sdk.md)

## 官方原文入口

1. [Overview — SDK 5 分层架构](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/introduction/overview)
2. [Core Concepts — 五大原语与三阶段](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/getting-started/core-concepts)
3. [Temporal Contracts — 触发、选择、缺失策略](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/building-operators/temporal-contracts)
4. [Relevant Technologies — CUDA/HoloMQ/GXF 迁移](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/introduction/relevant-technologies)
5. [SDK Installation — 平台及依赖](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/setup/sdk-installation)
6. [Slow/Fast Fusion — 可运行多速率示例](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/examples/slow-fast-fusion)
7. [Camera + IMU Control — 外部 Topic 示例](https://docs.nvidia.com/holoscan/sdk-user-guide/5-0-latest/examples/camera-imu-control)
8. [Migrating from 4.x](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/migrating-from-4x)

## 原文核对要点（非逐字转载）

- **产品定位：** Physical AI 实时传感器处理；主要处理不同速率传感器、融合与有界延迟的感知到输出链路。
- **分层：** HoloMQ（iceoryx2 的跨进程通道）→ 原生 Graph/Operator Runtime → Inference、Vision、Visualization 模块。传感器/机器人专用模块属于后续规划，不应写作 SDK 5.0 已交付。
- **核心编程模型：** `Graph` 构图 → `compile(graph)` 校验并产生可检查 `ExecutionPlan` → `run(plan)` 或 `run_async(plan)`。Operator 通过 `setup()`、`contract()`、`compute()` 分离结构、时序与计算。
- **时间约束：** `OnClock`/`OnEach`/`OnWindow` 等触发；`LatestWithin`/`BracketedWindow` 等选择；`Skip`/`SafeDefault`/`ReuseLast` 等缺失策略。时序契约是输入采样/调度语义，不等于自动保证执行完成时限。
- **代码示例的实际数字：** `slow_fast_fusion` 20 ms 与 2 ms 定时源、100 ms 最大慢源年龄；`camera_imu_control` 的模拟相机为 **20 ms（50 Hz）**，每帧 64 字节 Host Tensor；官方教程导言提及 60 Hz 相机是概念场景，不应把它误记为此源码的频率。
- **升级差异：** 4.x 的 GXF/UCX fragment 不是 5.0 原生运行时基础；旧版代码不可原样迁移。5.0 的外部 Topic/Channel 部分仍标记 Prototype。
- **运行环境：** Ubuntu 24.04、CUDA 13、Python ≥3.12、CMake ≥4.0，GPU Ampere 或更新；JetPack 7 / IGX OS 2，明确不支持 JetPack 6。
- **核对边界：** 官方示例为教学级合成输入，未证明真实硬件上的机器人闭环端到端安全、抖动或最坏时延。不能把 TensorRT/GPUDirect 可用性理解为每种主机默认具备零拷贝 RDMA 链路。

## 开源状态

- **SDK：已开源**，源码与可运行示例见 [源码归档](../repos/nvidia_holoscan_sdk.md)；官方 GitHub `LICENSE.txt` 为许可权威来源，第三方模块和可选依赖须分别核对许可证。
- **与 Sensor Bridge 的边界：** SDK 是运行时和算子框架；[Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md) 是 FPGA/Ethernet 传感器接入链路，独立软件与硬件依赖，不能视为 SDK 本体。
