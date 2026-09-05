# NVIDIA Warp 官方文档（stable）

> 来源归档

- **标题：** NVIDIA Warp Documentation
- **类型：** site（官方文档）
- **来源：** nvidia.github.io/warp
- **链接：** https://nvidia.github.io/warp/stable/
- **版本快照：** 1.17.0（2026-09-05）
- **代码：** <https://github.com/NVIDIA/warp>
- **入库日期：** 2026-09-05
- **一句话说明：** Warp 1.17 权威入口：定位、Quickstart、`warp-lang` 安装、示例画廊、语言 / API / 领域模块与可微说明。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-warp.md`](../../wiki/entities/nvidia-warp.md)

## 开源边界（步骤 2.5）

文档 Footer / 导航指向 [NVIDIA/warp](https://github.com/NVIDIA/warp)。安装页给 PyPI、conda-forge、GitHub Releases CUDA 13 wheel 与源码构建。→ **已开源**。

## 页面要点（2026-09-05）

### 定位（首页）

Warp is a Python framework for GPU-accelerated simulation, robotics, and machine learning. Warp takes regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Kernels are differentiable and can be used as part of machine-learning pipelines with PyTorch, JAX and Paddle.

### Quickstart

```text
pip install warp-lang
python -m warp.examples.<example_subdir>.<example>
```

可选依赖：`pip install warp-lang[examples]`。

### 安装页摘要

| 项 | 要求 |
|----|------|
| Python | 3.10+ |
| 包名 | `warp-lang` |
| 平台 | Win/Linux x86-64（+ Linux aarch64）：CPU + CUDA；macOS：仅 CPU |
| CUDA 12 驱动 | 525+ |
| CUDA 13 驱动 | 580+ |
| Nightly | NVIDIA Package Index，CUDA 12，无 macOS |
| Tile / MathDx | 完整支持建议 CUDA **12.6.3+** |

驱动不足：初始化警告，CUDA 设备不可用，CPU 仍可用。

### 文档结构（导航）

User Guide、Language Reference、API Reference、Domain Modules；另有 Contributing、Publications、Changelog。可微专页：<https://nvidia.github.io/warp/user_guide/differentiability.html>（MJWarp README 亦链此页并声明自身尚未接通）。

## 对 wiki 的映射

- 实体页：[`wiki/entities/nvidia-warp.md`](../../wiki/entities/nvidia-warp.md)
- 仓库：[`sources/repos/nvidia-warp.md`](../repos/nvidia-warp.md)
- 下游：[`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)、[`wiki/entities/mujoco-warp.md`](../../wiki/entities/mujoco-warp.md)
