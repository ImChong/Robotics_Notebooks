# nvidia-holoscan/holoscan-sdk — 官方源码

> NVIDIA 一手仓库与 `v5.0.0` 源码归档，核对日期：2026-11-12。

- **标题：** Holoscan SDK
- **类型：** repo
- **组织：** nvidia-holoscan
- **链接：** <https://github.com/nvidia-holoscan/holoscan-sdk>
- **版本：** [v5.0.0](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v5.0.0)（2026-10-21 发布；固定 tag 避免 main 分支漂移）
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sdk>
- **文档：** <https://docs.nvidia.com/holoscan/sdk-user-guide/>
- **入库日期：** 2026-11-12
- **核对范围：** 通读 v5.0.0 的 `examples/slow_fast_fusion/README.md`、`examples/slow_fast_fusion/cpp/slow_fast_fusion.cpp`、`examples/camera_imu_control/README.md`、`examples/camera_imu_control/cpp/camera_imu_control.cpp`，核对编译、图连接、时间策略、通道与停止路径；未对整个 SDK 仓库进行逐文件源码审计。
- **通读：** 已完成
- **沉淀到 wiki：** [NVIDIA Holoscan SDK](../../wiki/entities/nvidia-holoscan-sdk.md)
- **官方文档归档：** [Holoscan SDK 5](../sites/nvidia-holoscan-sdk-5.md)

## 源码定位与运行入口

| 路径（固定 v5.0.0） | 已核实作用 |
|---|---|
| [`examples/slow_fast_fusion/README.md`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0/examples/slow_fast_fusion/README.md) | `./slow_fast_fusion` / `--plan-json` / `--emit-artifacts` 的使用说明 |
| [`examples/slow_fast_fusion/cpp/slow_fast_fusion.cpp`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0/examples/slow_fast_fusion/cpp/slow_fast_fusion.cpp) | `main` 建 `Graph`，实例化 `PeriodicSource`、`FuseLatest`、`BoundedSink`；`compile`→`run_async`→`session.wait()` |
| [`examples/camera_imu_control/README.md`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0/examples/camera_imu_control/README.md) | `./camera_imu_control`、独立 `sensor_daemon` / `result_recorder` 和冷启动部署制品 |
| [`examples/camera_imu_control/cpp/camera_imu_control.cpp`](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0/examples/camera_imu_control/cpp/camera_imu_control.cpp) | `CameraImuGraph` 嵌入 `FusionGraph`，`import_topic` / `export_topic`，`WindowFusion` 的 `OnWindow+BracketedWindow`，`run_reference(plan)` 以 `ChannelRuntime` 做 smoke test |

## 源码级数据流

1. **构图：** `slow_fast_fusion.cpp:main` 用 `graph.op<PeriodicSource>` 注册慢源和快源，通过 `graph.add_flow` 连接 `FuseLatest` 与 `BoundedSink`，绑定 `RealtimeClock`。
2. **校验：** `compile(graph)` 编译为 `ExecutionPlan`；`plan.ok()` 为假则输出 `plan.json()` 并退出；`--plan-json` 可不启动运行时检查计划。
3. **采集：** 两个 `PeriodicSource::compute` 根据 `OnClock(20ms / 2ms)` 发出带采集时间戳的 `Sample<uint32_t>`。
4. **融合：** `FuseLatest::contract` 用 `OnEach{fast}` 驱动，并选 `LatestWithin{100ms}` 的慢源；缺数据则 `Skip`；`compute` 合并数值、转发快源元数据。
5. **结束：** `BoundedSink` 记录第六个结果时调用 `request_completion()`；`session.wait()` 后检查 `RunTermination::kRequestedCompleted`。这是整次运行完成请求，不代表所有排队下游数据已经 drain。
6. **进阶：** `camera_imu_control.cpp` 中 `CameraSource` 20 ms 产生 64-byte 模拟 Host Tensor，`WindowFusion` 用 `OnWindow` / `BracketedWindow` 接入外部 `ChannelRuntime` 发布的 IMU 整数，输出控制整数；`run_reference` 对通道连接、数据发布、接收、停止分别校验。

## 复现边界

- **已开源/可编译示例：** 是（示例路径与 CMake 已核对）；**本次没有本地 CUDA/SDK 环境编译或跑测试**。
- **硬件真实数据：** 两个示例使用数值与模拟 Tensor，不等同接入真实相机/IMU；外部独立进程必须遵守 catalog/时间戳要求。
- **版本差异：** 只把本页 API/时序图适用于 **5.0.0**；4.x 以 GXF 为基础且接口明显不同。
