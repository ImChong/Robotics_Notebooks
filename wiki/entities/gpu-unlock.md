---
type: entity
title: GPU Unlock
project_id: gpu-unlock
project: https://github.com/Zagitta/gpu-unlock
code: https://github.com/Zagitta/gpu-unlock
date: 2026-10-25
tags: [gpu, cuda, nvidia, performance, linux]
status: complete
summary: "GPU Unlock 是一个在 NVIDIA 驱动初始化前设置 SM speed-select 的 Linux 内核模块；Ampere 上游和 RTX 4090 Ada 移植均报告特定 FP16/BF16 指令吞吐提升，但不代表整应用等比例加速。"
updated: 2026-10-25
related:
  - ./nvidia-cuda.md
  - ./tensorrt.md
sources:
  - ../../sources/sites/gpu-unlock.md
  - ../../sources/repos/zagitta-gpu-unlock.md
  - ../../sources/repos/justaresearcher-gpu-unlock-ada.md
---

# GPU Unlock

**GPU Unlock** 是一组 Linux 内核模块项目，在 NVIDIA 驱动接管 GPU 前调整特定 SM speed-select 配置，以改变部分张量指令的发射速率。截图中的 RTX 30 系列上游与 RTX 4090 Ada 移植属于两个仓库，结果不能合并为一个普遍性能承诺。

## 一句话定义

在驱动初始化前设置 GPU speed-select 寄存器，以尝试恢复部分 FP16/BF16 运算路径的指令吞吐。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SM | Streaming Multiprocessor | GPU 计算单元；该工具涉及其 speed-select 配置 |
| FP16 | 16-bit Floating Point | 部分基准所测的低精度矩阵输入格式 |
| BF16 | Brain Floating Point 16 | 与 FP16 类似的 16 位浮点格式，指数范围更大 |
| TF32 | TensorFloat-32 | Ampere 上 Tensor Core 支持的矩阵运算格式 |
| FLR | Function Level Reset | PCIe 设备级功能复位；上游模块在驱动绑定前执行 |

## 为什么值得关注

消费级显卡的实际表现不只由芯片峰值决定，某些输入/累加精度组合的指令发射速率也可能影响矩阵运算。GPU Unlock 提供了可审阅的 Linux 内核模块和用户基准，展示驱动加载前的寄存器配置对特定设备和工作负载的影响。

## 核心原理

两个仓库的共同流程是，在标准 NVIDIA 驱动初始化前运行特权内核模块，识别目标 PCI 设备，并尝试设置 speed-select 配置；随后通过复位与读回等步骤确认状态，再由未修改的 NVIDIA 驱动初始化 GPU。

- **Ampere 上游（Zagitta）**：README 说明其适用 GA102、GA103、GA104、GA106 的指定设备 ID，设置 SS0/SS1 后执行 FLR。仓库公布的 RTX 3090 Ti 8192³ GEMM 测试中，FP16 87.46→118.39、BF16 87.50→157.41、TF32 42.58→82.61 TFLOPS；数据不是统一适用于所有 RTX 30 卡的增幅。
- **Ada 移植（JustAResearcher）**：README 报告一张 RTX 4090 的 FP16/BF16 输入、FP32 累加指令吞吐约从 165 TF/s 到 331 TF/s。独立测试和报告同时显示，实际 GEMM 的收益随矩阵尺寸变化，并未支持“所有推理吞吐翻倍”。

## 工程实践

评估时分别记录显卡 PCI ID、驱动版本、内核版本、功耗上限、时钟状态、输入/累加精度和矩阵形状；对照官方仓库的检查项确认模块确实应用到目标卡，再用相同基准做多轮前后测量。指令级微基准、GEMM、完整推理应用是不同层级，不能互相替代。

## 局限与风险

- 性能收益依卡型、设备固件、功耗限制、精度和算子形状而变；部分 Ampere 报告显示模块流程完成但 BF16 仍处于半速。
- Ada 仓库明确指出 RTX 4090 核心收益来自有限硬件测试，尚不构成长期稳定性或广泛应用验证；4090D、4070 Ti SUPER、4060 Ti 16GB 等路径具有各自实验性边界。
- 这是以高权限内核模块和 GPU 寄存器配置为基础的非官方工具。运行前应审阅代码、确认设备与内核兼容性，并准备恢复启动配置。
- 图中“3090 算力暴涨 80%、4090 吞吐翻倍”省略了卡型、精度、功耗和测试层级。4090 指令发射率约 2 倍不能直接改写成 vLLM 或其他推理引擎普遍提速 2 倍。

## 关联页面

- [NVIDIA CUDA](./nvidia-cuda.md) — GPU 计算与软件栈基础
- [TensorRT](./tensorrt.md) — NVIDIA 推理运行时；应用性能需单独实测

## 参考来源

- [GPU Unlock 官方项目入口与主张核对](../../sources/sites/gpu-unlock.md)
- [Zagitta/gpu-unlock 上游仓库](../../sources/repos/zagitta-gpu-unlock.md)
- [JustAResearcher/gpu-unlock-ada Ada 移植](../../sources/repos/justaresearcher-gpu-unlock-ada.md)

## 推荐继续阅读

- [Ampere 上游 README 与兼容性报告](https://github.com/Zagitta/gpu-unlock)
- [Ada / RTX 4090 实测说明与限制](https://github.com/JustAResearcher/gpu-unlock-ada)
