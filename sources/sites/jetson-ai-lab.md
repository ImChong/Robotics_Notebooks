# Jetson AI Lab（教程站）

> 来源归档

- **标题：** Jetson AI Lab — Tutorials
- **类型：** site（NVIDIA 官方边缘 AI 教程聚合）
- **链接：** <https://www.jetson-ai-lab.com/tutorials/>
- **入门教程：** <https://www.jetson-ai-lab.com/tutorials/getting-started-with-jetson/>
- **容器索引：** <https://www.jetson-ai-lab.com/>（Browse All Jetson Containers）
- **入库日期：** 2026-09-06
- **版本：** 站点 **2.0**（curated tutorials；旧内容见 archive）
- **一句话说明：** Jetson 上部署 **LLM/VLM/VLA** 与机器人应用的 **分步教程 hub**：Getting Started、GenAI、Gemma/Cosmos Reason、GR00T/OpenPi Thor、Agent Skills、优化与 GTC workshop。
- **沉淀到 wiki：** [`wiki/entities/jetson-ai-lab.md`](../../wiki/entities/jetson-ai-lab.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **站点** | NVIDIA 维护的 **教程与示例聚合**；多数教程链到 **开源模型/容器/仓库** |
| **硬件** | 需 **Jetson 开发套件/模组** + **JetPack**（商业 SDK，见 [`nvidia-jetpack.md`](./nvidia-jetpack.md)） |
| **Agent Skills** | **Jetson Device Skills**（设备侧）+ **Jetson BSP Skills**（刷机前主机侧）— 可复用工作流文档 |

## Getting Started with Jetson（2026-09-06 摘录）

推荐新开发者路径：

1. 按模组代际完成 **官方 Quick Start**（Thor / AGX Orin / Orin Nano — docs.nvidia.com）
2. **SSH 远程开发**（Orin USB-C 固定地址 `192.168.55.1` 可首连）
3. **VS Code / Cursor Remote-SSH** 连 Jetson 编辑与终端
4. 配置 **Jetson Agent Skills**（Device + BSP catalogs）
5. 后续：**SSD+Docker**、**RAM Optimization**、**Introduction to GenAI**

## 教程分区（Tutorials 索引，2026-09-06）

| 分区 | 数量 | 代表教程 |
|------|------|----------|
| **Getting Started** | 6 | Introduction to Jetson；Getting Started；**Agent Skills**；Yocto Quick Start (JP 7.2)；SSD+Docker；RAM Optimization |
| **Fundamentals** | 3 | GenAI on Jetson (Ollama/vLLM)；GenAI Benchmarking；Ollama on Jetson |
| **VLM** | 2 | **Gemma 4** (vLLM/llama.cpp)；**Cosmos Reason2** + Live VLM WebUI |
| **VLA** | 2 | **OpenPi π₀.₅** on Thor (TRT NVFP4)；**Isaac GR00T 1.7** on Thor |
| **Applications** | 6 | Multi-Modal AI Studio；NanoOWL；Live VLM WebUI；OpenClaw；NemoClaw；Reachy Mini assistant |
| **Model Optimization** | 5 | Fine-tune LLMs；**TensorRT Edge-LLM**；Model Optimizer NVFP4；Speculative Decoding；Unsloth |
| **Workshops** | 3 | **GTC 2026** Thor LLM/VLM；GTC DC 2025 production deployment；Hackathon Guide |

Featured 机器人/VLA 读法：VLA 分区直接对接 [Isaac GR00T](../../wiki/entities/isaac-gr00t.md) 与 Physical Intelligence OpenPi 机载部署叙事。

## 对 wiki 的映射

- 实体：[`wiki/entities/jetson-ai-lab.md`](../../wiki/entities/jetson-ai-lab.md)
- 平台：[`wiki/entities/nvidia-jetson.md`](../../wiki/entities/nvidia-jetson.md)
- JetPack：[`sources/sites/nvidia-jetpack.md`](./nvidia-jetpack.md)
- 开发者指南：[`sources/sites/jetson-linux-r392-developer-guide.md`](./jetson-linux-r392-developer-guide.md)
