# NVIDIA Holoscan Sensor Bridge（产品页）

> 来源归档

- **标题：** NVIDIA Holoscan Sensor Bridge
- **类型：** site（NVIDIA 官方产品/技术页）
- **链接：** <https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/>
- **文档：** <https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/introduction>
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **入库日期：** 2026-09-06
- **一句话说明：** **Sensor-over-Ethernet** 平台：FPGA 采集相机/雷达/LiDAR/RF 等传感器，经标准 API 与开源栈 **低延迟流式写入 GPU 内存**，面向医疗、机器人、仪器与信号处理等 mission-critical 边缘 AI。
- **沉淀到 wiki：** [`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **主机软件** | **已开源** — [nvidia-holoscan/holoscan-sensor-bridge](https://github.com/nvidia-holoscan/holoscan-sensor-bridge)（Apache-2.0） |
| **FPGA IP** | 产品页提供 **Holoscan Sensor Bridge IP** 与 User Guide；HoloHub 链到 **FPGA IP Source**（硬件侧另计许可） |
| **硬件** | 需 **HSB FPGA 参考设计/评估板** 或生态伙伴量产方案（Lattice / Microchip / Altera 等） |

## Overview 摘录（2026-09-06）

- **问题：** 多类传感器常需 **专有协议** 与定制驱动，集成/维护/扩展成本高。
- **方案：** HSB 提供 **标准 API + 开源软件**，经 **FPGA 接口** 将高速传感器数据 **直接流式写入 GPU 内存**。
- **架构：** Sensor → FPGA（HSB device）→ Ethernet（UDP）→ Host（ConnectX SmartNIC **GPUDirect** 或 Linux socket）→ Holoscan pipeline → GPU 算子（Bayer→RGBA、ISP、推理、可视化）。

## Highlights（官方标称）

| 指标 | 数值 | 备注 |
|------|------|------|
| **4K60 相机端到端延迟** | **17 ms** | photon→display，NVIDIA IGX Orin 上测 |
| **GPUDirect 信号处理** | **<1 ms** | IGX Orin + GPUDirect |
| **Camera-over-Ethernet CPU 占用** | **~1%** | AGX Thor 上 CoE 加速 |

## Benefits 摘录

- **超低延迟：** 相对传统系统宣称最高 **10×** 延迟降低。
- **易用：** 标准 API 覆盖 streaming DMA、GPIO/SPI/I2C/自定义寄存器、传输抽象层、SmartNIC 加速与 Linux socket；驱动开发时间宣称最高 **100×** 缩短。
- **可扩展：** 模块化 **10GbE–100GbE**；软件定义传感器配置。
- **安全：** 端到端安全协议，最高 **SIL 2** 等级。

## 生态与入门路径

| 路径 | 说明 |
|------|------|
| **软件** | Holoscan Sensor Bridge IP + User Guide；在标准 API 上构建驱动与 Holoscan 应用 |
| **评估板** | FPGA 伙伴（Altera / Lattice / Microchip PolarFire）；MCU 伙伴（NXP / STMicro） |
| **量产传感器** | 相机、模拟信号转换器等集成方案（Sensor Partners 页） |

## 对 wiki 的映射

- 实体：[`wiki/entities/holoscan-sensor-bridge.md`](../../wiki/entities/holoscan-sensor-bridge.md)
- 平台：[`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)（AGX Orin / Thor 主机）
- 代码：[`sources/repos/nvidia_holoscan_sensor_bridge.md`](../repos/nvidia_holoscan_sensor_bridge.md)
- 文档 intro：[`sources/sites/holoscan-sensor-bridge-docs-intro.md`](./holoscan-sensor-bridge-docs-intro.md)
