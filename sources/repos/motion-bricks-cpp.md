# motion-bricks.cpp — 原始资料归档

- **来源**：https://github.com/localai-org/motion-bricks.cpp
- **类型**：repo
- **机构**：LocalAI（GitHub `localai-org`；权重托管 Hugging Face `LocalAI-io`）
- **归档日期**：2026-09-15
- **上游**：NVIDIA [MotionBricks](https://nvlabs.github.io/motionbricks/) / [GR00T-WholeBodyControl/motionbricks](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks) / [arXiv:2604.24833](https://arxiv.org/abs/2604.24833)
- **许可**：移植代码 **Apache-2.0**；GGML 子模块与模型权重各保留原许可
- **开源结论（步骤 2.5）**：**已开源** — GitHub 公开 C++/GGML 推理仓（无独立 `*.github.io` 项目页，以仓库 README / `docs/` 为入口）；G1 F32 GGUF（约 **0.73 GB**、**183,148,382** 学习参数）与 15 种上游 style 已发布；可选 **GGML SONIC + MuJoCo 3.12** 物理跟踪与 Kimodo 动画回放
- **快照**：约 210 stars；C++23 为主

## 一句话说明

**motion-bricks.cpp** 把 NVIDIA [MotionBricks](../../wiki/methods/motionbricks.md) 的 **G1 batch-one 推理路径** 移植到 **C++23 / GGML**：CPU 或 Vulkan 上加载原生 GGUF，从运动/朝向/风格命令规划 root 与 pose token，经 VQ 解码与特征转换输出 **34 关节** 骨架运动；不依赖 Python/PyTorch 运行时。

## 为什么值得保留

- 官方 MotionBricks 预览代码在 [GR00T-WholeBodyControl/motionbricks](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)（Python + Git LFS 权重）；本仓把 **实时生成层** 拆成 **C ABI + PureGo 绑定**，适合游戏引擎、嵌入式或无 Python 机器人栈
- **端到端 G1 推理已闭环**：严格 GGUF 加载、root/duration 规划、pose-token 预测、VQ 解码、418/414/413 特征转换、style 对齐与骨骼动画输出；CPU/Vulkan 在参考套件上 **duration 与 pose-token 决策一致**
- **可选物理跟踪**：Demo 可接 **GGML SONIC** + **MuJoCo 3.12**，展示 target/reference/physical 三骨架；并支持 **Kimodo** 动画回放（Kimodo GLB→`.mbstyle` 转换仍为后续集成）
- 机器人选型上它是 MotionBricks **部署档**：能出 `mb_agent_plan` 运动缓冲与 Three.js/WebSocket 流式 Demo，但 **Smart Object 全谱、Kimodo 风格直转、量化权重** 等尚未完全移植

## 能力边界（README，截至 2026-09）

已实现：G1 F32 GGUF 校验加载、safetensors→GGUF 转换、stateful `mb_agent`（movement/facing/style → plan/advance）、15 种上游 style `.mbstyle`、CPU/Vulkan 开放环 parity（14 plans）、Go/Three.js 交互 Demo、可选 SONIC CPU 推理（Ryzen 7900 上单核 encoder+decoder **~0.93 ms**）。

未实现 / 后续：Kimodo GLB 直转 `.mbstyle`；除 G1 batch-one 外的完整 Smart Primitives 作者工具链；量化 GGUF。

## 模型与权重

| 产物 | 位置 | 说明 |
|------|------|------|
| G1 F32 运行时包 | [MotionBricks-G1-GGML](https://huggingface.co/LocalAI-io/MotionBricks-G1-GGML) | NVIDIA Open Model License；manifest + SHA-256 校验 |
| 上游 checkpoint 来源 | [GR00T-WholeBodyControl/motionbricks](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/a0732b642c0333077e127a2f56ab0014c196bca4/motionbricks) | NVIDIA 当前通过 Git LFS 分发，非独立 HF 模型仓 |
| 15 种 style | `scripts/convert_styles.py` 从上游 safe 目录转换 | 运行时 `.mbstyle` 资产 |

默认构建会下载并校验 HF commit `cc2a47603dbc203a4f18f35dd06ed3611833f506`；离线构建用 `-DMOTIONBRICKS_DOWNLOAD_MODELS=OFF`。

## 运行时接口（工程入口）

| 入口 | 位置 / 命令 |
|------|-------------|
| C ABI | `mb_model_load` / `mb_agent_create` / `mb_agent_plan` / `mb_agent_advance`；输出 row-major F32 root `[frames,3]` 与 local XYZW `[frames,34,4]` |
| 设备 | CPU / Vulkan；SONIC 可选 `sonic-cpu` preset 与多 variant CPU 后端 |
| 构建 | C++23 + CMake 3.25+ + Ninja；`cmake --preset debug`；可选 Nix flake |
| 权重 | `python scripts/download_gguf_weights.py`；`motionbricks-cli inspect` 校验参数计数 |
| Demo | `go run ./demo`（PureGo，**CGO_ENABLED=0**）；WebSocket 流式 + Three.js 骨架 |
| SONIC+MuJoCo | 见 `docs/SONIC-GGML.md`、`docs/MUJOCO-UPGRADE.md` |

GGML 以 **pinned git submodule** 引入；`-DMOTIONBRICKS_ENABLE_GGML=OFF` 可仅构建非神经 ABI 子集。

## 对 wiki 的映射

1. **[motion-bricks.cpp（实体页）](../../wiki/entities/motion-bricks-cpp.md)** — 本地 GGML 运行时、SONIC/MuJoCo 可选栈与缺失能力
2. **[MotionBricks（方法页）](../../wiki/methods/motionbricks.md)** — 上游 Smart Primitives 与 GR00T 角色
3. **[Kimodo（上游实体）](../../wiki/entities/kimodo.md)** — Demo 可选 Kimodo 动画回放来源
4. **[SONIC 运动跟踪](../methods/sonic-motion-tracking.md)** — 可选 GGML 物理跟踪后端

## 关联原始资料

- [MotionBricks 项目页](../sites/motionbricks-project.md)
- [MotionBricks 论文摘录](../papers/motionbricks.md)
- [GR00T-WholeBodyControl](./gr00t_wholebodycontrol.md)
- [kimodo.cpp（同生态 C++ 移植）](./kimodo-cpp.md)
