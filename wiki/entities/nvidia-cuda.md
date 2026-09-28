---
type: entity
title: NVIDIA CUDA
date: 2026-09-28
tags: [framework, gpu, cuda, nvidia, deployment, jetson]
status: complete
summary: "NVIDIA CUDA 是 GPU 并行计算平台与工具链：编程模型、NVCC/runtime、数学库与 Nsight；TensorRT、Warp、Isaac 与 JetPack 机载栈均建立在其驱动+CTK 版本对齐之上。"
updated: 2026-09-28
related:
  - ./tensorrt.md
  - ./nvidia-warp.md
  - ./nvidia-jetson.md
  - ./curobo.md
  - ../comparisons/robot-policy-deployment-dev-board-selection.md
  - ../queries/vla-deployment-guide.md
sources:
  - ../../sources/sites/nvidia-cuda-toolkit-docs.md
  - ../../sources/repos/cuda-samples.md
---

# NVIDIA CUDA

**CUDA**（Compute Unified Device Architecture）是 NVIDIA 的 **GPU 并行计算平台与编程模型**：开发者用 CUDA C++/Python 等编写 **kernel**，经 **NVCC / NVRTC** 编译，由 **CUDA Runtime（cudart）** 在 GPU 上调度执行。机器人研究与工程中，往往不必手写大量 CUDA kernel，但 **驱动版本、JetPack 捆绑的 CTK、Stream/Graph 语义** 决定了 [TensorRT](./tensorrt.md)、[Warp](./nvidia-warp.md)、PyTorch、Isaac 等栈能否 **稳定低延迟** 跑在 Orin/Thor/dGPU 上。

## 一句话定义

**NVIDIA GPU 上的并行计算操作系统层**——上层框架共享同一套 driver + runtime + 数学库；版本错配会直接表现为 launch 失败、静默降速或机载抖动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 并行平台与 API |
| CTK | CUDA Toolkit | 编译器、runtime、库与 Nsight 工具包 |
| NVCC | NVIDIA CUDA Compiler | 主机端编译 CUDA C++ 的编译器驱动 |
| NVRTC | NVIDIA Runtime Compilation | 运行时编译 CUDA 源码 |
| cudart | CUDA Runtime API | 内存、stream、launch、Graph 等运行时 |
| CUPTI | CUDA Profiling Tools Interface | Nsight 等 profiler 的底层接口 |
| H2D / D2H | Host-to-Device / Device-to-Host | CPU 与 GPU 间数据搬运 |
| SM | Streaming Multiprocessor | GPU 上执行 kernel 的计算单元 |

## 为什么重要

- **机器人 NVIDIA 栈的公共地基**：[TensorRT](./tensorrt.md) engine、[cuRobo](./curobo.md) 运动规划、[NVIDIA Warp](./nvidia-warp.md) JIT kernel、Isaac Sim/Lab 与多数 PyTorch CUDA 扩展，最终都落到 **cudart + 驱动**。
- **机载延迟工具箱**：官方 [CUDA Graph](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) 把重复 launch 序列 capture 成单图，降低 **batch=1** 控制环开销——与 [VLA 部署指南](../queries/vla-deployment-guide.md) 中 **CUDA Graph** 优化叙事（如 [APXInf](./apxinf.md)、[ReflexVLA](./paper-reflexvla.md)）同向。
- **版本矩阵是工程门禁**：CTK **13.4** 起 Linux 上 **驱动不再随 Toolkit 捆绑**；13.x 与 12.x 的 **minor compatibility** 分界在驱动 **580** 附近（见 Release Notes）。Jetson 须以 [JetPack](./nvidia-jetson.md) 文档为准，不能假设与工作站 CTK 混装。

## 核心结构（官方编程模型）

1. **Host / Device 异构**：CPU 编排；GPU 执行并行 kernel（grid → block → thread）。
2. **内存空间**：Global、Shared、Constant、Local；机器人 pipeline 常见 **pinned host memory** + **async copy** 与计算 overlap（Best Practices Guide §10.1）。
3. **Stream**：同一 device 上操作队列；多 stream 可 overlap kernel 与 H2D/D2H（控制环与 RL 采样/学习并发时相关）。
4. **CUDA Graph**：capture 重复 subgraph，减少 launch 开销；适合固定拓扑的感知/策略推理环。
5. **库与工具链**：cuBLAS/cuFFT/cuSPARSE 等；**Nsight Systems / Nsight Compute** 做系统 timeline 与 kernel 级分析。

```mermaid
flowchart LR
  host[Host CPU<br/>ROS / Python 编排]
  h2d[Pinned H2D / Streams]
  kern[CUDA Kernels<br/>自定义 / 库 / TRT / Torch]
  d2h[D2H 或 device-only 输出]
  host --> h2d --> kern --> d2h
  kern -.->|重复环可 capture| graph[CUDA Graph]
  graph --> kern
```

## 工程实践

| 场景 | 做法 |
|------|------|
| **工作站训练 + Jetson 部署** | 训练侧 CTK 可以较新；机载以 **JetPack 锁定 CUDA** 为准，在目标板编译 TRT engine / 自定义扩展 |
| **查驱动是否够新** | 对照 [CUDA 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) 中 **Driver Branch / Min driver** 表；13.4 新特性需 **R615+** |
| **Linux 新装 CTK** | 13.4+ **单独安装驱动**；勿假设 runfile 仍捆绑 driver |
| **低延迟 VLA / 策略** | 固定 input shape → 考虑 **CUDA Graph** + Nsight 验证 launch gap；见 Best Practices「Assess → Deploy」 |
| **调试数值** | sim2real 敏感算子注意 **FP32 非结合律**（Best Practices §7.3） |
| **学习路径** | 读 Programming Guide Part 1–2 → 跑 [`cuda-samples`](https://github.com/NVIDIA/cuda-samples) 对照 → 用 Nsight  profile 自己的 ONNX/TRT 环 |

## 局限与风险

- **平台绑定**：CUDA 仅 NVIDIA GPU；跨厂商需 ORT/MNN/ROCm 等另一条栈（见 [UniLab](./unilab.md) 对非 CUDA 训练的叙事）。
- **「装了 PyTorch CUDA 版 = 环境完整」**：缺 **匹配驱动** 或 **CTK 头文件/版本** 时，自定义扩展编译或 TRT build 仍会失败。
- **CTK 与 JetPack 混用**：Thor **SBSA + CUDA 13** 与 Orin 安装器不同；须跟官方 Jetson 升级教程，而非仅 `apt install cuda`。
- **过度手写 kernel**：多数机器人团队应优先框架与 TRT；CUDA 文档价值在 **理解瓶颈与版本**，而非全面 kernel 开发。

## 关联页面

- [TensorRT](./tensorrt.md) — 建立在 CUDA 上的推理编译/runtime
- [NVIDIA Warp](./nvidia-warp.md) — Python JIT 到 CUDA kernel
- [NVIDIA Jetson](./nvidia-jetson.md) — JetPack 捆绑 CUDA 的机载平台
- [cuRobo](./curobo.md) — CUDA 端到端运动规划示例
- [机器人运控开发板选型（按策略网络模型）](../comparisons/robot-policy-deployment-dev-board-selection.md)
- [VLA 部署指南](../queries/vla-deployment-guide.md)

## 参考来源

- [NVIDIA CUDA Toolkit 官方文档（一手资料）](../../sources/sites/nvidia-cuda-toolkit-docs.md)
- [NVIDIA/cuda-samples（官方示例仓库）](../../sources/repos/cuda-samples.md)

## 推荐继续阅读

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA Toolkit 产品页与 Jetson 升级教程](https://developer.nvidia.com/cuda-toolkit)
