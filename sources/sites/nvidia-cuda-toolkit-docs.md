# NVIDIA CUDA Toolkit 官方文档（一手资料）

> 来源归档

- **标题：** CUDA Toolkit Documentation & Release Notes
- **类型：** site（NVIDIA 官方文档）
- **链接：** <https://docs.nvidia.com/cuda/>
- **产品入口：** <https://developer.nvidia.com/cuda-toolkit>
- **Release Notes 快照：** [CUDA 13.4 Update 1](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)（入库日 2026-09-28）
- **入库日期：** 2026-09-28
- **一句话说明：** CUDA 并行计算平台与工具链的权威文档：编程模型、编译器/runtime、数学库、Nsight 与 **驱动/CTK 版本兼容矩阵**（Jetson 与 dGPU 部署必读）。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-cuda.md`](../../wiki/entities/nvidia-cuda.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **CUDA Toolkit** | **可下载 SDK**（NVIDIA 许可），非整包 GitHub 开源；与 GPU 驱动配合使用 |
| **示例代码** | [`NVIDIA/cuda-samples`](https://github.com/NVIDIA/cuda-samples) — **已开源**（BSD-3），见 [`sources/repos/cuda-samples.md`](../repos/cuda-samples.md) |
| **容器** | NGC Catalog 提供 CUDA 基础镜像（非源码仓库） |

## 文档结构（docs.nvidia.com/cuda）

| 文档 | URL | 用途 |
|------|-----|------|
| **CUDA C++ Programming Guide** | <https://docs.nvidia.com/cuda/cuda-c-programming-guide/> | 编程模型、C++/Python GPU 编程、CUDA Graph、Unified Memory、多 GPU |
| **CUDA C++ Best Practices Guide** | <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/> | Assess → Parallelize → Optimize → Deploy；内存/occupancy/stream |
| **CUDA Toolkit Release Notes** | <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/> | 组件版本、驱动分支、minor compatibility、弃用 |
| **CUDA Compatibility Guide** | <https://docs.nvidia.com/cuda/cuda-compatibility/> | 跨驱动/跨 CTK 运行与升级 |
| **CUDA Installation Guide** | 各 OS 子页 | Linux / Windows / WSL / Jetson 安装路径 |

Programming Guide 五部分（官方组织，2026-09-28）：

1. **Introduction & Programming Model** — 与语言无关的 GPU 执行模型
2. **Programming GPUs in CUDA** — C++/Python 入门与常见性能要点
3. **Advanced CUDA** — 多 GPU 与进阶控制
4. **CUDA Features** — **CUDA Graph**、动态并行、图形互操作、Unified Memory 等专章
5. **Technical Appendices** — C++ 语言扩展与硬件规格

## Release Notes 13.4 Update 1 要点（机器人栈相关）

### 组件与工具（节选）

- **NVCC / cudart / NVRTC：** 13.4.92
- **cuBLAS** 13.8.0.4、**cuDNN**（随 JetPack/CTK 捆绑策略以 JP 文档为准）
- **CUPTI / Nsight Compute / Nsight Systems** — 与 kernel 级 profiling、系统级 timeline 对齐
- **Thrust / CUB / libcu++** 3.4.3
- **CUDA Compatibility Package (Orin)** — arm64-sbsa，便于 Orin 与 SBSA 栈对齐

### 驱动与 Toolkit 解耦

> The NVIDIA driver is no longer bundled with the CUDA Toolkit – on Windows starting with CUDA 13.1, and on Linux starting with CUDA 13.4.

须从 [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) 单独安装匹配驱动。

### 驱动分支与 minor compatibility（Release Notes 表）

| CUDA Toolkit | 对应驱动分支 |
|--------------|--------------|
| 13.4 | R615 |
| 13.3 | R610 |
| 13.2 | R595 |
| 13.1 | R590 |
| 13.0 | R580 |

- 已有 **CUDA 13.x** 应用可在 **驱动 ≥580** 上运行（minor version compatibility）；**13.4 新特性** 需 **R615+**。
- **12.x** 与 **13.x** 的 minor compatibility 分界：12.x 驱动范围 **≥525 且 <580**；13.x **≥580**。

### 平台与 Jetson

- 官方 Toolkit 页含 **「CUDA Upgrades for Jetson Devices」** 视频教程（developer.nvidia.com/cuda-toolkit → Tutorials）。
- 嵌入式侧 **JetPack** 捆绑 CUDA/TensorRT 版本；Thor 对齐 **SBSA + CUDA 13**（见 [`nvidia-jetpack.md`](./nvidia-jetpack.md)）。

## Best Practices Guide 摘要（部署环）

官方推荐流程：**Assess → Parallelize → Optimize → Deploy**。

机器人常见瓶颈对应章节：

| 现象 | 文档方向 |
|------|----------|
| 小 batch 推理 launch 开销大 | **CUDA Graph**（Programming Guide Part 4）；与 [ReflexVLA](../../wiki/entities/paper-reflexvla.md)、[APXInf](../../wiki/entities/apxinf.md) 叙事一致 |
| CPU 喂 GPU 抖动 | **Pinned memory**、**async H2D** 与计算 overlap（Best Practices §10.1） |
| RL 采样 vs 学习争抢 GPU | **Stream**、多 GPU（Advanced CUDA） |
| 数值与 sim2real | §7 精度、非结合律浮点 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/nvidia-cuda.md`](../../wiki/entities/nvidia-cuda.md)
- 示例仓库：[`sources/repos/cuda-samples.md`](../repos/cuda-samples.md)
- 下游栈：[TensorRT](../../wiki/entities/tensorrt.md)、[NVIDIA Warp](../../wiki/entities/nvidia-warp.md)、[JetPack / Jetson](../../wiki/entities/nvidia-jetson.md)
