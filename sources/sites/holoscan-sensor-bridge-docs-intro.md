# NVIDIA Holoscan Sensor Bridge 3.0 — 官方 Introduction / 架构 / 应用

> 来源归档；版本：3.0 GA；官方文档复核日期：2026-11-12。

- **标题：** Holoscan Sensor Bridge 3.0 Documentation
- **类型：** site（NVIDIA 官方技术文档）
- **链接：** <https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/introduction>
- **架构：** <https://docs.nvidia.com/holoscan/sensor-bridge/applications/architecture>
- **应用：** <https://docs.nvidia.com/holoscan/sensor-bridge/applications/applications>
- **性能测试：** <https://docs.nvidia.com/holoscan/sensor-bridge/applications/latency>
- **新设备驱动：** <https://docs.nvidia.com/holoscan/sensor-bridge/applications/new-sensors>
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **入库日期：** 2026-09-06
- **复核日期：** 2026-11-12
- **核对范围：** 通读 3.0 Introduction、Architecture、Applications 中 IMX274 player 数据流与调度部分、Latency、新传感器驱动的核心段落，并与对应源码 `MIGRATION.md`、`examples/imx274_player.py`、`examples/linux_imx274_player.py` 交叉检查；不声称通读整套 API Reference。
- **通读：** 已完成
- **一句话说明：** Holoscan Sensor Bridge 3.0 运行在 Holoscan SDK 5.0 上，以类型化双端口、编译执行图与 RoCE/CoE/socket 模式组织实时传感器接入。
- **沉淀到 wiki：** [Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md)

## 版本及主机要求（3.0）

- **Holoscan SDK：** 3.0 已移植到 SDK **5.0.0**；旧 4.x `Application/compose` 不兼容，运行时转为 `Graph → compile → run_async`。
- **软件：** Ubuntu 24.04、CUDA 13、NVIDIA Driver R580+、Python 3.12+；容器外编译需要 CMake 4.0+。
- **平台：** Jetson AGX Orin/Thor + JetPack 7.2.1、IGX Thor + IGX SW 2.0.1、DGX Spark + DGX OS 7.6.0、符合要求的 Linux GPU workstation。
- **网络：** Orin 开发板 Linux socket；Thor 支持 socket / CoE；IGX Thor 和 DGX Spark 的 ConnectX 支持 RoCE 或 socket。
- **不支持：** SDK 5 不支持旧 IGX Orin（IGX OS 1）或 JetPack 6；这些平台需使用 [HSB 2.7](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/2.7.0)。

## 控制面与数据面

1. 设备 FPGA 收集 CSI-2/其他原始数据并以 UDP 发送到主机；板侧 I2C、SPI、GPIO 控制由用户态 **Sensor object** 与 `DataChannel` 处理。
2. `Enumerator.find_channel` 根据局域网广播找到传感器接口；`DataChannel` 管理接收地址和网络配置。
3. `NativeImx274Sensor` 配置时钟、传感器模式、寄存器；接收 operator 的 `device_start` / `device_stop` 回调负责启停视频。
4. 有 ConnectX 时 `RoceReceiverOp` 使用 RDMA/GPU 通路；普通 Linux 网络用 `LinuxReceiverOp` 经 CPU 接收与复制；另有 Thor CoE 路径。
5. SDK 5 中接收算子输出两个类型化端口：**`tensor`**（像素数据）和 **`metadata`**（`hololink::CaptureMetadataT`）；下游需要按采集时间对齐它们，不是 4.x 的单个 GXF Entity 和无类型 metadata dictionary。
6. `CsiToBayerOp` → `ImageProcessorOp` → `BayerDemosaicOp` → `HolovizOp`；对接推理任务可以增加 Inference 与叠加绘制节点。
7. Python 的 IMX274 示例用 `graph.add_flow` 接口，对显示队列使用 depth=1 的 `DROP_OLDEST`，以减少慢显示造成的历史帧积压。

## 端到端复现与工程边界

- **示例启动：** 参考 [Applications](https://docs.nvidia.com/holoscan/sensor-bridge/applications/applications)；实际 `imx274_player.py` 先初始化 CUDA/网卡/传感器，再构建图、`compile`、`run_async`。数据流的具体接收端取决于是否存在 ConnectX。
- **性能测量：** 官方 [Latency](https://docs.nvidia.com/holoscan/sensor-bridge/applications/latency) 区分 FPGA 首包时间、帧结束、主机 CPU 唤醒、下游算子起始与 GPU 显示交接；SDK 5 的 `complete` 不是窗口真正呈现时间，因此不能把它解释成 photon-to-display。
- **已知代价：** 源码 `MIGRATION.md` 明确 HSDK5 receiver 由 RDMA buffer 到 Tensor pool **每帧有一次 cudaMemcpyAsync D2D 拷贝**；不能无条件宣称应用内全流程严格零拷贝。
- **版本局限：** `ecam0m30tof_player.py` 在 3.0 文档中仍标记「尚不支持 SDK 5」，部分旧分布式示例已移除。

## 对 wiki 的映射

- [Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md)
- [Holoscan SDK 5.0](../../wiki/entities/nvidia-holoscan-sdk.md)
- [产品页归档](./nvidia-holoscan-sensor-bridge.md)
- [源码归档](../repos/nvidia_holoscan_sensor_bridge.md)
