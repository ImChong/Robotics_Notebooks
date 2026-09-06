# Holoscan Sensor Bridge — Introduction（官方文档）

> 来源归档

- **标题：** Introduction | Holoscan Sensor Bridge Documentation
- **类型：** site（NVIDIA 官方文档）
- **链接：** <https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/introduction>
- **上级文档：** <https://docs.nvidia.com/holoscan/sensor-bridge/>
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **入库日期：** 2026-09-06
- **一句话说明：** HSB **主机侧软件模型**：FPGA 经 UDP/Ethernet 送数；ConnectX **RoCE/GPUDirect** 直写 GPU；Holoscan **operator 管线** + **sensor object** 驱动 IMX274 等示例。
- **沉淀到 wiki：** [`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)

## Overview 摘录

- **数据路径：** 外设 → HSB device **FPGA** → **UDP over Ethernet** → 主机；**IGX / DGX Spark** 上 **ConnectX SmartNIC** 可将 UDP **直接写入 GPU 内存**。
- **Holoscan 集成：** 主机软件提供 operator，把 HSB 网络数据接入 Holoscan pipeline；示例含 **Sony IMX274** 相机的视频处理与推理。
- **支持主机：** **IGX Devkit**、**Jetson AGX Orin**、**Jetson AGX Thor**、**DGX Spark**。
- **无 SmartNIC 主机：** 如 **Jetson AGX Orin Devkit** 走 **Linux socket** Ethernet（性能受 OS 网络栈限制）。
- **PTP：** 支持硬件 **PTP 时间戳**（IGX 与 AGX Orin 板载网口）；用于接收时刻、管线延迟测量与传感器同步。

## Software 模型

- **Holoscan 应用：** 用 operator 列表 + `add_flow` 连接输入输出，配置 pipeline 调度。
- **网络接收 operator：** 例 **`RoceReceiverOp`** — 从 HSB 源收 UDP 并写入 GPU 内存；相机场景下为 **CSI-2 Bayer** 内存块。
- **现成算子：** Bayer→RGBA、ISP、**推理**、可视化、数据完整性测试 — 面向 **实时视频** 处理链。
- **可定制：** operator 以 **源码** 提供；可 fork 适配非视频场景（如 5G 天线高速 analog）；可 upstream 回 NVIDIA。

## Applications 与 Sensor objects

- **HoloscanApplication：** 子类重写 `configure()` 构建 pipeline — 典型链：**采集 → 处理 → 输出 → 下发**。
- **Sensor object：** 设备级配置/监控 API（例 **`Imx274Cam`**：初始化、曝光 `set_exposure`、健康监测）；相当于 **用户态驱动**，不含应用逻辑。
- **应用层 operator** 可调 sensor API（如自动曝光 operator 读帧后调 `set_exposure`）。

## Host requirements

- 任意 **NVIDIA Holoscan 支持** 的系统均可运行；**最佳性能** 推荐 **IGX + ConnectX SmartNIC**。
- 无 ConnectX 时（如 Jetson AGX Orin）性能受 **主机 OS 网络栈** 限制。

## 对 wiki 的映射

- 实体：[`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)
- 产品页：[`sources/sites/nvidia-holoscan-sensor-bridge.md`](./nvidia-holoscan-sensor-bridge.md)
- 代码：[`sources/repos/nvidia_holoscan_sensor_bridge.md`](../repos/nvidia_holoscan_sensor_bridge.md)
