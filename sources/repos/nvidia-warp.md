# NVIDIA/warp

> 来源归档

- **标题：** NVIDIA Warp
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/warp>
- **文档：** <https://nvidia.github.io/warp/stable/>
- **产品页：** <https://developer.nvidia.com/warp-python>
- **PyPI：** `warp-lang`
- **Stars：** ~7.1k（2026-09-05）
- **入库日期：** 2026-09-05
- **许可证：** Apache-2.0（代码）；源码构建会再拉 libmathdx（NVIDIA SLA，不随本仓条款覆盖）
- **一句话说明：** 把普通 Python 函数 JIT 成 CPU/GPU kernel 的可微计算框架，面向仿真、机器人与机器学习；Newton / MuJoCo Warp 的计算底座。
- **沉淀到 wiki：** 是 → [`wiki/entities/nvidia-warp.md`](../../wiki/entities/nvidia-warp.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0） |
| **代码** | <https://github.com/NVIDIA/warp> |
| **文档** | <https://nvidia.github.io/warp/stable/>（当前 **1.17.0**） |
| **安装** | `pip install warp-lang`；示例 `pip install warp-lang[examples]` |
| **Omniverse** | Kit 扩展注册表 `omni.warp.core` |

文档站与 GitHub README 互指。macOS wheel **仅 CPU**，无 Metal。PyPI 默认 CUDA 12.9 runtime；CUDA 13 wheel 在 GitHub Releases。

## README / 文档要点（2026-09-05）

- `@wp.kernel` + `wp.launch`：Python 函数 JIT 到 CPU 或 CUDA。
- 核可微，可接入 **PyTorch / JAX / Paddle**。
- 原语覆盖物理仿真、几何、FEM、tile 编程；示例分 `core` / `fem` / `optim` / `tile`。
- Python **3.10+**。Windows/Linux：CPU + CUDA；macOS Apple Silicon：仅 CPU。
- CUDA 12.x 包要求驱动 **525+**；CUDA 13.x 要求 **580+**。驱动不够时 CUDA 设备不可用，仍可走 CPU。
- Nightly：`pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/`（CUDA 12，无 macOS）。
- `warp.sim` **已弃用**；[Newton](https://github.com/newton-physics/newton) 扩展并接替该模块（见 Newton README）。

冒烟（README）：百万粒子引力，约 20 行 `@wp.kernel` + `wp.launch`。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| JIT / 可微 / 安装面 | [`wiki/entities/nvidia-warp.md`](../../wiki/entities/nvidia-warp.md) |
| 机器人物理引擎层 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)、[`wiki/entities/mujoco-warp.md`](../../wiki/entities/mujoco-warp.md) |
| 文档站 | [`sources/sites/nvidia-warp-docs.md`](../sites/nvidia-warp-docs.md) |
