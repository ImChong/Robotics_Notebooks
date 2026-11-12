# nvidia-holoscan/holoscan-sensor-bridge — 官方源码（3.0）

> NVIDIA 官方 GitHub 源码归档；核对日期：2026-11-12。

- **标题：** Holoscan Sensor Bridge
- **类型：** repo
- **组织：** nvidia-holoscan
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **文档：** <https://docs.nvidia.com/holoscan/sensor-bridge/>
- **版本：** 默认分支 `VERSION=3.0.0`；对应 [3.0-GA Release Notes](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/RELEASE_NOTES.md)（2026 年 11 月）
- **入库日期：** 2026-09-06
- **复核日期：** 2026-11-12
- **核对范围：** 通读当前仓库 `README.md`、`VERSION`、`RELEASE_NOTES.md` 的 3.0-GA 部分、`MIGRATION.md` 中 SDK 4→5 核心迁移规则，以及 `examples/imx274_player.py`、`examples/linux_imx274_player.py` 与 `examples/imx274_player.cpp` 的入口和完整数据流；不宣称审计全仓每个算子或所有 C++ 底层实现。
- **通读：** 已完成
- **一句话说明：** 公开的 Hololink/HSB 主机代码，涵盖传感器控制、UDP/RoCE/Linux 接收算子、Python/C++ IMX274 示例与 SDK 5 迁移说明。
- **沉淀到 wiki：** [Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md)
- **官方产品归档：** [HSB 产品页](../sites/nvidia-holoscan-sensor-bridge.md)
- **官方技术归档：** [HSB 3.0 Introduction](../sites/holoscan-sensor-bridge-docs-intro.md)

## 开源和版本边界

| 项 | 核实 |
|---|---|
| SDK/主机端代码 | **已开源**；仓库 README 明确 Apache-2.0，引用 [LICENSE](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/LICENSE) |
| HSB 3.0 的依赖 | Holoscan SDK **5.0.0+**，Docker `docker/build.sh` 默认安装 5.0.0；可 `--hsdk=VERSION` 指定 |
| FPGA 相关资源 | README 提供 [FPGA IP Source](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/hsb/fpga_ip/)；授权范围应以具体代码包和 [THIRD_PARTY_NOTICES.md](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/THIRD_PARTY_NOTICES.md) 为准，含部分受限制的第三方 RTL |
| 旧主机 | JetPack 6 与 IGX OS 1 使用 [2.7.0 分支](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/tree/2.7.0)，不可直接用 3.0 |
| 运行验证 | **未在本次环境完成硬件运行**：需要相应 FPGA/相机及支持的 GPU 网络路径 |

## 已阅读的真实运行入口

| 路径 | 已核实的作用 |
|---|---|
| [`README.md`](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/README.md) | `docker/build.sh` / `docker/demo.sh`，HSDK5 编译与许可证边界 |
| [`MIGRATION.md`](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/MIGRATION.md) | `Graph`、类型化端口、`CaptureMetadataT`、`CompileOptions`、接收 buffer→Tensor 池复制与新的 `frame_limit` |
| [`examples/imx274_player.py`](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/examples/imx274_player.py) | `main()`→`run()`；枚举 HSB、配置相机、`RoceReceiverOp`→`CsiToBayerOp`→`ImageProcessorOp`→`BayerDemosaicOp`→`HolovizOp` |
| [`examples/linux_imx274_player.py`](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/examples/linux_imx274_player.py) | 同一图中改用 `LinuxReceiverOp`；上游 `DROP_OLDEST` 防止慢处理时队列堆积 |
| [`examples/imx274_player.cpp`](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/main/examples/imx274_player.cpp) | C++ 由 `graph.op<>`、`set_params()`、`compile()` 和 `run_async()` 构造/启动；与 Python 同属 SDK5 |

## Python `imx274_player.py` 已核对的时序

1. **设备发现：** `hl.find_channel(hololink_ip)` → `hl.DataChannel(metadata)` → `hololink_dev.start()/reset()`。
2. **相机参数：** `hl.NativeImx274Sensor` → `setup_clock()` → `configure(camera_mode)`，可设置测试图案；默认 Python mode 0 为 4K60（C++ 示例默认是 1080p60）。
3. **图构建：** 用 `hs.Graph` 建 `RoceReceiverOp`、`CsiToBayerOp`、`ImageProcessorOp`、`BayerDemosaicOp`、`HolovizOp`；receiver 带 `device_start=camera.start`、`device_stop=camera.stop`。
4. **两路输出：** receiver 和前两个图像算子有 `tensor` + `metadata` 端口；`graph.add_flow` 按端口连接；末端显示边为 `DROP_OLDEST` depth=1。
5. **编译：** `CompileOptions.deployment.bind_tensor_output_device` 将各 GPU 输出绑定设备；`plan=hs.compile(graph, options)`，校验 `plan.ok`。
6. **运行：** `with hs.run_async(plan) as session: session.wait()`，接收 operator 生命周期回调启停相机；正常退出 `hololink_dev.stop()` 并释放 CUDA primary context。
7. **关闭风险：** Python 此示例没捕获 `KeyboardInterrupt`，直接 Ctrl+C 可能跳过显式 `hololink_dev.stop()`；C++ player 通过 `player_support.hpp` 的信号管理实现协作式停止。
8. **复制语义：** `MIGRATION.md` 指出 GPU RDMA buffer 到 SDK5 Tensor pool **每帧 1 次 `cudaMemcpyAsync` Device-to-Device**，故不能把此示例宣传为端到端绝对零拷贝。

## 复现入口

官方 `README.md`：
```bash
docker/build.sh
docker/demo.sh
```
容器内运行示例前仍需按 [Hardware Setup](https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/hardware-setup) 连接 HSB 板与相机并设网卡；运行参数见 `python3 examples/imx274_player.py --help` 或 `python3 examples/linux_imx274_player.py --help`（实际目录以容器内的官方工作目录为准）。

## 对 wiki 的映射

- [HSB 技术节点](../../wiki/entities/holoscan-sensor-bridge.md)
- [Holoscan SDK 5 节点](../../wiki/entities/nvidia-holoscan-sdk.md)
- [产品页](../sites/nvidia-holoscan-sensor-bridge.md)
- [官方文档](../sites/holoscan-sensor-bridge-docs-intro.md)
