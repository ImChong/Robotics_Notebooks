# NVIDIA JetPack SDK

> 来源归档

- **标题：** NVIDIA JetPack
- **类型：** site（官方 SDK 产品页）
- **链接：** <https://developer.nvidia.com/embedded/jetpack>
- **下载：** <https://developer.nvidia.com/embedded/jetpack/downloads>
- **配套文档：** [Jetson Linux Developer Guide r39.2](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/)
- **入库日期：** 2026-09-06
- **当前主线：** **JetPack 7**（页内强调 **7.2** 能力：Yocto、NemoClaw 一键安装、Agent Skills）
- **一句话说明：** Jetson 官方 **BSP + AI 栈捆绑**：Linux 6.8 / Ubuntu 24.04、CUDA/TensorRT/vLLM、Isaac ROS、**Agentic**（NemoClaw + Jetson Agent Skills），Thor 对齐 **SBSA + CUDA 13**。
- **沉淀到 wiki：** [`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **JetPack SDK** | **商业可下载 SDK**（非整仓 GitHub 开源）；组件含 CUDA/TensorRT 等 NVIDIA 许可 |
| **Jetson Linux** | 内核/BSP/刷机工具 — 随 JetPack 分发；定制见 Developer Guide |
| **Yocto/OE4T** | **7.2 起官方支持** Yocto Project；预构建 demo-image-full 配方（AI Lab 有 Quick Start） |
| **Agent Skills / NemoClaw** | 文档与安装器公开；**agentic 工作流**随 JetPack 7.2 推广 |

## JetPack 7 要点（2026-09-06）

### 平台与基础

- 全量支持 **Jetson Orin** 与 **Jetson Thor**
- **Linux kernel 6.8** + **Ubuntu 24.04 LTS**
- **Preemptable RT kernel**、**MIG**、**Holoscan Sensor Bridge**
- 模块化、**cloud-native**（容器/K8s 微服务）
- **Jetson Thor → SBSA** 对齐；**CUDA 13.0** 统一 Arm 目标安装（Thor 用 **SBSA installer**）

### Agentic（7.2）

| 组件 | 作用 |
|------|------|
| **NemoClaw** | 单命令安装；本地+云端模型编排 |
| **Jetson Agent Skills** | Device 侧（诊断、LLM serving、编解码…）+ BSP 侧（pinmux、PCIe、camera…） |

### AI Compute Stack

CUDA · cuDNN · **TensorRT** · PyTorch · **vLLM** · **SGLang** · **Triton Inference Server**

### Jetson Linux 组件

Flashing（SDK Manager）· Security（Secure Boot、fTPM、OTA）· Graphics · **Multimedia APIs** · OpenCV/VisionWorks · **Yocto Project**

### 配套 SDK

**DeepStream** · **Isaac ROS** · **Holoscan SDK**

### 社区入口

[**Jetson AI Lab**](https://www.jetson-ai-lab.com/tutorials/) · Developer Forums

## 对 wiki 的映射

- 平台：[`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)
- 教程 hub：[`wiki/entities/jetson-ai-lab.md`](../../wiki/entities/jetson-ai-lab.md)
- 开发者指南：[`sources/sites/jetson-linux-r392-developer-guide.md`](./jetson-linux-r392-developer-guide.md)
- Isaac ROS：[`wiki/entities/isaac-ros-visual-slam.md`](../../wiki/entities/isaac-ros-visual-slam.md) 等
- TensorRT：[`wiki/entities/tensorrt.md`](../../wiki/entities/tensorrt.md)
