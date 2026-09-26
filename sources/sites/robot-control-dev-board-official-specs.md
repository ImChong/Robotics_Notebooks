# 机器人运控 / 机载推理开发板官方规格汇总

> 来源归档

- **标题：** 机器人运控与机载策略推理常见开发板官方规格（Jetson Orin / Thor、RDK X5 / S100、RK3588、Raspberry Pi 5）
- **类型：** site / hardware（多厂商产品页归档）
- **入库日期：** 2026-09-26
- **一句话说明：** 把人形 / 四足 / 小型足式机器人部署策略网络时最常见的几块开发板的 **官方标称规格**（CPU、AI 加速器、内存带宽、功耗、实时核与总线）集中归档，供 [按策略模型选开发板](../../wiki/comparisons/robot-policy-deployment-dev-board-selection.md) 对比页引用。
- **开源状态：** 不适用（商业硬件产品页；软件栈与模型库另见各厂商开发者站）
- **沉淀到 wiki：** [`wiki/comparisons/robot-policy-deployment-dev-board-selection.md`](../../wiki/comparisons/robot-policy-deployment-dev-board-selection.md)

## 原始链接

| 厂商 | 产品 | 官方页面 |
|------|------|----------|
| 英伟达（NVIDIA） | Jetson Orin 模组（Nano / NX / AGX） | <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/> |
| 英伟达（NVIDIA） | Jetson Thor（T4000 / T5000） | <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/> |
| 地瓜机器人（D-Robotics） | RDK X5 | <https://en.d-robotics.cc/rdkx5> |
| 地瓜机器人（D-Robotics） | RDK S100 / S100P | <https://en.d-robotics.cc/rdks100> |
| 瑞芯微（Rockchip） | RK3588 SoC | <https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html> |
| 瑞莎（Radxa） | CM5 模组（RK3588S2） | <https://radxa.com/products/cm/cm5/> |
| 树莓派（Raspberry Pi） | Raspberry Pi 5 | <https://www.raspberrypi.com/products/raspberry-pi-5/> |
| 宇树（Unitree） | G1 规格表（算力段） | <https://www.unitree.com/g1/> |

## 规格摘录（2026-09-26 抓取）

### NVIDIA Jetson Orin 模组

| 模组 | AI 算力（稀疏 INT8） | GPU | CPU | 内存 / 带宽 | 功耗 |
|------|---------------------|-----|-----|-------------|------|
| AGX Orin 64GB | 275 TOPS | 2048-core Ampere + 64 Tensor Core | 12× Cortex-A78AE @ 2.2 GHz | 64 GB LPDDR5 / 204.8 GB/s | 15–60 W |
| AGX Orin 32GB | 248 TOPS | 1792-core Ampere + 56 Tensor Core | 8× A78AE @ 2.0 GHz | 32 GB LPDDR5 / 204.8 GB/s | 15–60 W |
| Orin NX 16GB | 157 TOPS | 1024-core Ampere + 32 Tensor Core | 8× A78AE @ 2.0 GHz | 16 GB LPDDR5 / 102.4 GB/s | 10–40 W |
| Orin NX 8GB | 117 TOPS | 512-core Ampere + 16 Tensor Core | 6× A78AE @ 1.7 GHz | 8 GB LPDDR5 / 102.4 GB/s | 10–40 W |
| Orin Nano 8GB | 67 TOPS | 1024-core Ampere + 32 Tensor Core | 6× A78AE @ 1.7 GHz | 8 GB LPDDR5 / 102 GB/s | 7–25 W |
| Orin Nano 4GB | 34 TOPS | 512-core Ampere + 16 Tensor Core | 6× A78AE @ 1.7 GHz | 4 GB LPDDR5 / 51 GB/s | 7–25 W |

- 官方页功耗写作「NX 16GB/8GB: 10W–25W/40W」「Nano 8GB/4GB: 7W–15W/25W」，上限对应 Super / MAXN 模式；本表合并为区间。
- **Orin Nano Super Developer Kit** 定价 **$249**、67 TOPS、25 W 模式（NVIDIA 2024-12 发布，经 JetsonHacks 等转载：<https://jetsonhacks.com/2024/12/17/jetson-orin-nano-super-developer-kit/>）。

### NVIDIA Jetson Thor

| 模组 | AI 算力 | GPU | CPU | 内存 / 带宽 | 功耗 |
|------|---------|-----|-----|-------------|------|
| Jetson T5000 | 2070 TFLOPS（FP4 稀疏） | 2560-core Blackwell，第 5 代 Tensor Core，MIG 10 TPC | 14-core Neoverse-V3AE | 128 GB LPDDR5X 256-bit / 273 GB/s | 40–130 W |
| Jetson T4000 | 1200 TFLOPS（FP4 稀疏） | 1536-core Blackwell，MIG 6 TPC | 12-core Neoverse-V3AE | 64 GB LPDDR5X 256-bit / 273 GB/s | 40–70 W |

### 地瓜机器人 RDK X5

- **BPU：** 10 TOPS；**CPU：** 8× Cortex-A55 @ 1.5 GHz；**GPU：** 32 GFLOPS
- **内存：** 4 GB / 8 GB LPDDR4；Micro SD 存储
- **机器人接口：** **1× CAN FD**；2× 4-lane MIPI CSI；1× 千兆网口（PoE）；4× USB 3.0；28 GPIO（可复用 UART / PWM / I2C / SPI）
- **供电：** 5 V / 5 A；**系统：** Ubuntu 22.04

### 地瓜机器人 RDK S100 / S100P

| 项 | RDK S100 | RDK S100P |
|----|----------|-----------|
| BPU（INT8 等效） | 80 TOPS | 128 TOPS |
| CPU | 6× Cortex-A78AE @ 1.5 GHz | 6× Cortex-A78AE @ 2.0 GHz |
| MCU（实时核） | 4× Cortex-R52+ @ 1.2 GHz | 同左 |
| GPU | Mali-G78AE 100 GFLOPS | 同左 |
| 内存 | 12 GB LPDDR5（96-bit） | 24 GB LPDDR5（96-bit） |
| 存储 | 64 GB eMMC + M.2 Key M（PCIe 3.0 x1） | 同左 |
| 网络 / USB | 2× 千兆网口；4× USB 3.0 | 同左 |
| 供电 | 12–20 V DC | 同左 |

- 官方把 S100 描述为 **「一板双脑」**：A78AE 跑感知 / 策略推理，R52+ MCU 跑实时运控。官方页 **未列 CAN**（抓取时核对）。
- 第三方评测将 S100P 定位为 Orin NX 16GB 的国产替代（CNX Software 2026-08-31：<https://www.cnx-software.com/2026/08/31/d-robotics-rdk-s100p-a-128-tops-alternative-to-nvidia-jetson-orin-nx-16gb-with-cortex-a78ae-r52-cores/>）。

### 瑞芯微 RK3588

- **CPU：** 4× Cortex-A76 + 4× Cortex-A55；**工艺：** 8 nm
- **GPU：** Mali-G610 MC4（OpenGL ES / Vulkan 1.2 / OpenCL）
- **NPU：** **6 TOPS**，三核，支持 INT4 / INT8 / INT16 / FP16 / BF16 / TF32
- **I/O：** PCIe 3.0 / 2.0、USB 3.1、SATA 3.0、千兆以太网（RGMII）、8K 编解码
- **Radxa CM5**（RK3588S2 模组，56×41 mm）：同为 4× A76 + 4× A55、Mali-G610 MP4、6 TOPS NPU；输入 3.6–5.2 V、典型 2 A；官方承诺供货至 2032-09（<https://radxa.com/products/cm/cm5/>、<https://docs.radxa.com/en/som/cm/cm5>）。
- 内存容量、功耗、价格 **由具体板卡决定**（如 Radxa Rock 5 / CM5、Orange Pi 5 等），SoC 官方页不给出。

### Raspberry Pi 5

- **CPU：** Broadcom BCM2712，4× Cortex-A76 @ 2.4 GHz；**GPU：** VideoCore VII
- **内存：** 1 / 2 / 4 / 8 / 16 GB LPDDR4X-4267；**无 NPU**
- **扩展：** PCIe 2.0 x1（需 M.2 HAT）
- **供电：** 5 V / 5 A USB-C PD（官方推荐 27 W 电源）
- **价格：** 16 GB 版 **$305**（官方页）

### Unitree G1（整机算力段，供对照）

- G1 与 G1 EDU 均含 **「8 核高性能 CPU」** 作为基础运控算力；G1 EDU 可选 **高算力模组**（官方写作「Orin 等多品牌型号可选」），**未给 TOPS**。

## 推理工具链与模型文件格式（2026-09-26 抓取）

| 平台 | 官方工具链 / runtime | 部署文件格式 | 输入格式 | 量化 | 出处 |
|------|---------------------|--------------|----------|------|------|
| 瑞芯微 RK3588 等 | **RKNN-Toolkit2**（PC 端转换）→ 板端 **RKNN Runtime**（C/C++，`librknnrt.so`）或 **RKNN-Toolkit-Lite2**（Python） | **`.rknn`** | ONNX、PyTorch、TensorFlow 等（model zoo 示例以 ONNX 为主） | model zoo 列 **INT8 / FP16** 两档；v2.3.2 起有自动混合精度 | <https://github.com/airockchip/rknn-toolkit2>、<https://github.com/airockchip/rknn_model_zoo> |
| 地瓜 RDK X5 | **OpenExplorer** Docker 工具链：`hb_mapper makertbin --model-type onnx` | **`.bin`** | ONNX 为主 | **INT8** PTQ（需校准集） | <https://github.com/D-Robotics/rdk_model_zoo> |
| 地瓜 RDK S100 / S100P | OpenExplorer（S 系列，HBDK 编译） | **`.bin`（PTQ）/ `.hbm`（QAT）**——官方 FAQ 原文「`.bin` for PTQ, `.hbm` for QAT」；`rdk_model_zoo_s` README 写部署 `*.bin`；CNX 评测称 S 系列原生格式为 `.hbm`。**口径不一，以所用 OpenExplorer 版本文档为准** | ONNX 为主 | PTQ / QAT | <https://developer.d-robotics.cc/rdk_doc/en/rdk_s/FAQ/toolchain/>、<https://github.com/D-Robotics/rdk_model_zoo_s> |

- RDK 官方 FAQ：**超出 BPU 约束（如 CxHxW > 8192）的算子回落 CPU 计算**；「少量算子受影响且整体性能达标则无需处理」。
- rdk_model_zoo_s README：精度异常时先确认 **OpenExplorer Docker 与板端 `libdnn.so` 均为最新发布版本**。
- rdk_model_zoo FAQ：**即便是纯 BPU 模型，输入 / 输出的量化 / 反量化节点也在 CPU 上执行**。
- NVIDIA TensorRT（`.onnx` → 目标 GPU 专属 engine / plan）、ONNX Runtime、OpenVINO、ncnn、LiteRT 的格式说明已在本库对应实体页与 [ORT vs MNN vs TensorRT](../../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md) 覆盖，此处不重复。

## 为什么值得保留

- 本库已有 [NVIDIA Jetson](../../wiki/entities/nvidia-jetson.md)、[Jetson Orin NX](../../wiki/entities/jetson-orin-nx.md)、[人形「大脑」选型](../../wiki/entities/open-source-humanoid-brains.md) 等页，但 **国产板（RDK / RK3588）与树莓派没有官方规格来源**，且缺少「按策略模型类型选板」的统一视角。
- 规格集中后，wiki 对比页可以把 **机载实测延迟**（PredActor、APXInf、VLA-ULAP 等已入库资料）与 **官方标称** 放在同一张表里，避免只比 TOPS。

## 对 wiki 的映射

- [机器人运控开发板选型（按策略网络模型）](../../wiki/comparisons/robot-policy-deployment-dev-board-selection.md) — 主落点
- [NVIDIA Jetson](../../wiki/entities/nvidia-jetson.md) — Jetson 产品线（Orin / Thor）已有摘要
- [开源人形机器人「大脑」选型](../../wiki/entities/open-source-humanoid-brains.md) — x86 / Jetson / 国产平台三分法
