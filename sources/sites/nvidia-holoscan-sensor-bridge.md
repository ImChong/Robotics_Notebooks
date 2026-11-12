# NVIDIA Holoscan Sensor Bridge（官方产品页，3.0）

> 来源归档；2026-11-12 重新核对 NVIDIA 官方产品原文，保留旧版本数据的口径说明。

- **标题：** NVIDIA Holoscan Sensor Bridge
- **类型：** site（NVIDIA 官方产品/技术页）
- **链接：** <https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/>
- **文档：** <https://docs.nvidia.com/holoscan/sensor-bridge/getting-started/introduction>
- **代码：** <https://github.com/nvidia-holoscan/holoscan-sensor-bridge>
- **入库日期：** 2026-09-06
- **复核日期：** 2026-11-12
- **核对范围：** 通读产品页 Overview、Highlights（含测试平台脚注）、Benefits、Starting Options、Sensor Partners 和开放 IP 入口；并对照当前 3.0 官方 Introduction 与仓库 README。仅称已通读这些主要原文，不包括所有下游伙伴网站。
- **通读：** 已完成
- **一句话说明：** FPGA/MCU + UDP/Ethernet 的高速传感器接入；加速网卡可将数据写入 GPU，Linux socket 为非加速备用路径。
- **沉淀到 wiki：** [Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md)

## 当前官方产品页要点（2026-11-12）

- **应用：** 传感器数据进入 GPU，适用于机器人、医疗与仪器系统；兼容相机、雷达、LiDAR、超声与射频前端。
- **架构：** 传感器 → HSB FPGA/MCU → UDP over Ethernet → 主机网络接收 → GPU / Holoscan 实时算子处理。
- **三种网络主机路径：** ConnectX 主机使用 **RoCEv2**；Jetson Thor 可使用 **Camera-over-Ethernet (CoE)**；其他主机可走 **Linux sockets**。
- **外设控制面：** 软件 API 覆盖 GPIO、SPI、I2C、寄存器和 streaming DMA；接收数据面与设备控制面应分别建模。
- **可扩展性：** 产品页描述从 1GbE 到 100GbE 的部署；具体带宽仍受传感器、FPGA 及主机网络配置约束。

## 性能宣称及严格适用边界

| 指标 | 产品页当前数字 | 对应测试条件／注意事项 |
|---|---|---|
| 平均 sensor-to-GPU latency | **10 µs** | Lattice CertusPro-NX → IGX T7000；不是整机 photon-to-display |
| Worst-case jitter | **44 µs** | 与以上 10 µs 组对应的产品页脚注，不可直接挪用为机器人全栈最坏时延 |
| CPU 利用率 | **<1%** | IGX T5000 + FuSa Receiver，**并非**前两项同一测试配置 |
| 产品层收益 | 最高 **10×** 时延改善、**100×** 驱动开发加速 | NVIDIA 的营销声明，不是对任意机器人系统的通用实测 |
| 安全 | 声称端到端安全协议可满足 **up to SIL 2** | 不意味着引用此技术的整机自动获得 SIL 2 认证 |

**历史口径说明：** 本仓旧版归档曾列 4K60 photon-to-display 17 ms、GPUDirect 信号处理 <1 ms 和 ~1% CPU；这些与当前产品页的 10 µs/44 µs/<1% 测试配置不同，不能合并为一组相同条件的性能指标。做横向比较必须核对测量边界和硬件型号。

## 开源与硬件边界

| 项目 | 已核实状态 |
|---|---|
| 主机软件 | [nvidia-holoscan/holoscan-sensor-bridge](https://github.com/nvidia-holoscan/holoscan-sensor-bridge)，Apache-2.0；3.0 接入 Holoscan SDK 5.0 |
| FPGA IP | 当前官方产品页明确称 **open-source FPGA IP**，源码下载入口见其 `Holoscan Sensor Bridge IP`；实际设计包/第三方器件 IP 许可须逐项核对，不应继续写成全部 FPGA IP 都是商业闭源 |
| MCU firmware | 产品页称与开源 FPGA IP 一起提供，具体文件和板级授权须看对应包 |
| 硬件 | Lattice / Microchip / Altera 等评估板与合作伙伴量产模组需另行采购/集成 |

## 对 wiki 的映射

- [Holoscan Sensor Bridge](../../wiki/entities/holoscan-sensor-bridge.md)
- [NVIDIA Holoscan SDK 5.0](../../wiki/entities/nvidia-holoscan-sdk.md)
- [官方架构介绍](./holoscan-sensor-bridge-docs-intro.md)
- [官方源码及运行入口](../repos/nvidia_holoscan_sensor_bridge.md)
