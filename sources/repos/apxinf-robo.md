# APXinf-robo（VLA 端侧推理与 OpenPI 兼容服务）

> 来源归档

- **标题：** APXinf-robo — Edge VLA Inference for Embodied Robots
- **类型：** repo
- **组织：** RLinf 生态；引擎子模块 [infinigence/ApxInf](https://github.com/infinigence/ApxInf)（无问芯穹）
- **代码：** <https://github.com/RLinf/APXinf-robo>
- **引擎文档：** <https://github.com/infinigence/ApxInf>（porting / kernels / model API）
- **许可：** Apache-2.0
- **入库日期：** 2026-09-16
- **复核日期：** 2026-11-07（raw.githubusercontent.com 上 RLinf/APXinf-robo `main` 的 README 全文）
- **核对范围：** README 全文（快速开始、Python API、OpenPI 兼容服务、性能表、精度说明、LIBERO 评测流程、基准层级、构建环境）；引擎子模块 infinigence/ApxInf 的源码与文档、`scripts/` 与 `doc/` 目录、`bench_pi05.py` 实现均**未读**，所有延迟数字出自 README 自述，本次没有在硬件上复现。
- **通读：** 已完成
- **一句话说明：** 面向 **Jetson Thor/Orin** 与 RTX 的 **VLA 端侧推理引擎** 封装：Rust 核心 + Python `build_robot_policy` / OpenPI websocket serve；首发 **π₀.₅**（BF16/FP8/INT8）；LIBERO-10 评测与 `bench_pi05.py` _latency 工具齐全。
- **步骤 2.5：** **已开源**（2026-09-16 GitHub 复核）— 完整构建说明、eval-libero、Agent model-port skill；checkpoint 不随仓分发，需 `lerobot/pi05_libero_base` 等外部权重。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [APXInf 实体](../../wiki/entities/apxinf.md) | wiki 主节点 |
| [RLinf](rlinf.md) | 同生态：训练 → APXInf 端侧部署 |
| [OpenPI / π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md) | 默认 benchmark 与 OpenPI 协议兼容 |
| [VLA 部署指南](../../wiki/queries/vla-deployment-guide.md) | 专用推理引擎 vs 通用 vLLM/TensorRT 路线 |
| [NVIDIA Jetson](../../wiki/entities/nvidia-jetson.md) | Thor `sm_110` / Orin `sm_87` 为主要验证平台 |

## 能力摘要（README）

- **L1–L3 API：** `load_bare_model` → `build_robot_policy` → `websocket_server`（OpenPI wire protocol）
- **Robot presets：** `franka_libero`、`unitree_g1` 等
- **精度：** BF16（全平台）；FP8（Thor，需 calibration）；INT8 W8A8（Orin/Ada）
- **评测：** `apxinf-robo eval-libero`；π₀.₅ reference **92.4%**，Thor FP8 **92.2%**（500 trials）
- **Thor FP8 P50：** **41.16 ms**（10 flow steps）；**26.32 ms**（onestep 剪枝）；双视角 224×224

## 为何值得保留

- 填补 **「RLinf 训练栈 → 机载 π₀.₅ 低抖动推理」** 的公开工程入口，与 FluxVLA（LimX 训练 DevOps）形成不同厂商的端侧对照。
- **OpenPI-compatible serve** 降低已有 openpi-client 机器人栈的迁移成本。

## 2026-11-07 README 复核补充

- 性能表口径：π₀.₅，两路视图 224×224 NHWC uint8，10 个 flow step，H=10，batch 1，延迟为稳态 CUDA Graph 回放的 **P50**。**Jetson AGX Thor：BF16 72.45 ms（13.8 Hz）、FP8 41.16 ms（24.3 Hz）；Jetson AGX Orin：BF16 165.67 ms（6.0 Hz）；RTX 4090：BF16 31.38 ms、INT8 25.99 ms。** README 写作「Jetson AGX Thor」，没有标 T4000/T5000，亦没有标 Orin 的显存规格。
- onestep 动作生成剪枝（集成 STEP，arXiv 2602.08245）：AGX Thor BF16 44.05 ms、FP8 **26.32 ms**；AGX Orin BF16 119.05 ms（8.4 Hz）；RTX 4090 BF16 20.36 ms。
- LIBERO-10（10 任务×50 回合，H=10，replan=5，seed 7；π₀.₅ 参考值 92.4%）：Thor BF16 464/500=92.8%，Thor FP8 461/500=92.2%，AGX Orin BF16 460/500=92.0%。
- 其他模型：π₀-FAST、GR00T N1.7（AGX Orin W8A8 1 视图 P50 56.7 ms、Thor FP8 32.6 ms）、Qwen-Drive（Thor BF16 482.6 ms）。
- **精度说明：** BF16 为默认且各设备可用；**FP8 仅 Thor，Orin 没有 FP8 Tensor Core、不支持**，FP8 需逐张量激活缩放，须用部署数据生成 `calibration.json`；INT8 为 W8A8，针对 Orin（SM87）与 Ada（SM89）优化。
- **构建：** 需完整 CUDA 工具链与 Rust；内核按本机 GPU 计算能力编译，须在部署机上构建，交叉编译须指定 `APXINF_CUDA_ARCH`（sm_87 Orin、sm_101 Thor-U、sm_110 Thor）；已验证：AGX Thor sm_110 + CUDA 13.0，AGX Orin sm_87 + CUDA 12.6 / 13.2，RTX 4090 sm_89 + CUDA 12.8。
