# APXinf-robo（VLA 端侧推理与 OpenPI 兼容服务）

> 来源归档

- **标题：** APXinf-robo — Edge VLA Inference for Embodied Robots
- **类型：** repo
- **组织：** RLinf 生态；引擎子模块 [infinigence/ApxInf](https://github.com/infinigence/ApxInf)（无问芯穹）
- **代码：** <https://github.com/RLinf/APXinf-robo>
- **引擎文档：** <https://github.com/infinigence/ApxInf>（porting / kernels / model API）
- **许可：** Apache-2.0
- **入库日期：** 2026-09-16
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
