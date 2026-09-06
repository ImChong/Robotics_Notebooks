# NVIDIA Jetson Embedded Systems（产品门户）

> 来源归档

- **标题：** NVIDIA Jetson for Next-Generation Robotics / Embedded Systems
- **类型：** site / 硬件产品线门户
- **URL：** https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
- **厂商：** 英伟达（NVIDIA）
- **入库日期：** 2026-09-06
- **一句话说明：** Jetson 嵌入式边缘 AI 计算平台总入口：从 **Jetson Thor** 到 **Orin / Xavier / TX2 / Nano** 模组谱系、**JetPack** 软件栈、Physical AI 推理运行时与合作伙伴生态；机器人机载算力与 [HIL](../../wiki/concepts/hardware-in-the-loop.md) 目标硬件的主线选型页。
- **开源状态：** **部分** — 硬件模组与 JetPack 为商业产品；开发者站提供 SDK/文档；**Jetson AI Lab** 聚合开源模型与示例；具体仓库以 developer.nvidia.com 为准。
- **沉淀到 wiki：** [`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)

---

## 平台定位（2026-09-06 抓取）

- **Slogan：** *Powering the Future of Embedded Edge AI*；*leading platform for robotics and edge AI*.
- **Physical AI：** 从 Orin 到 **Jetson Thor**，为机器人部署优化的 **physical AI hardware**；优化运行 **Nemotron、Cosmos、Isaac GR00T** 及社区开源模型。
- **软件栈：** **JetPack SDK**（预构建、agentic-ready、云原生边缘服务）+ **JetPack 7** + **Isaac 仿真框架** + 云集成；**Jetson AI Lab** 支持热门开源模型。

## Jetson Thor（新旗舰）

| 项 | 规格（Developer Kit / T5000 档） |
|----|----------------------------------|
| AI 算力 | 最高 **2070 FP4 TFLOPS**（sparse） |
| GPU | **Blackwell** 架构，2560 CUDA core + 96 第 5 代 Tensor Core |
| 内存 | **128 GB** LPDDR5X |
| 功耗 | **40–130 W** |
| vs AGX Orin | 官方称最高 **7.5×** AI 算力、**3.5×** 能效 |

模组形态：**Jetson T5000**、**Jetson T4000** 等；Developer Kit 含 1TB NVMe、QSFP 相机/网络等。

## 产品线速查（官方页面列举）

| 系列 | AI 性能（标称） | 功耗 | 外形参考 |
|------|-----------------|------|----------|
| **Jetson Thor** | 最高 2070 TOPS（FP4 sparse） | 40–130 W | 100×87 mm 模组 / DevKit 更大 |
| **Jetson AGX Orin** | 最高 275 TOPS | 15–60 W | 100×87 mm |
| **Jetson Orin NX** | 最高 157 TOPS | 10–25 W | 70×45 mm SO-DIMM |
| **Jetson Orin Nano** | 最高 67 TOPS | 7–15 W | 70×45 mm；入门 **$199** 起 |
| Jetson AGX Xavier | 最高 32 TOPS | 10–40 W | legacy |
| Jetson Xavier NX | 21 TOPS | 10–20 W | legacy |
| Jetson TX2 系列 | ~1.3 TFLOPS | 7.5–20 W | legacy |
| Jetson Nano | 0.5 TFLOPS | 5–10 W | legacy |

> 完整跨代对比表见原页 **Compare NVIDIA Jetson Module Specifications**（含 GPU/CPU/DLA/PVA、CSI、PCIe、视频编解码、I/O、功耗等）。

## 软件与生态要点

- **Jetson Software：** 统一支持全部 Jetson 模组；实时传感处理、视觉 AI、先进机器人特性；**agentic ready**（可在 Jetson 上构建/部署 AI agent）。
- **JetPack 7：** 与 Isaac 仿真框架、云集成，面向 generative AI 与 physical AI 快速迭代。
- **Jetson AI Lab：** 开源模型与 openclaws 等资源入口。
- **合作伙伴生态：** 边缘 appliance、工业 PC、机器人整机方案商。
- **NVIDIA IGX Platform：** 工业级、功能安全与企业软件支持的并行产品线（门户另链）。

## 入门路径（页面 CTA）

| 入口 | 用途 |
|------|------|
| [NVIDIA Developer Site](https://developer.nvidia.com/embedded-computing) | 软件与文档 |
| Jetson Developer Kits | 模组 / 开发套件购买 |
| Compare Specifications | 跨代选型 |
| Product Roadmap | 产品路线 |

## 对 wiki 的映射

- 平台实体：**`wiki/entities/nvidia-jetson.md`**
- 子型号深读：[jetson-orin-nx](../../wiki/entities/jetson-orin-nx.md)
- HIL 目标硬件：[hardware-in-the-loop](../../wiki/concepts/hardware-in-the-loop.md)
- Isaac 人形栈：[isaac-gr00t](../../wiki/entities/isaac-gr00t.md)（Thor 部署叙事）
- 机载推理对比：[onnxruntime-vs-mnn-vs-tensorrt](../../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)
