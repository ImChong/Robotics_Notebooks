# AMC（zjq19990202-spec/AMC）源码核查

> 来源归档（repo / 官方科研发布实现）

- **仓库：** <https://github.com/zjq19990202-spec/AMC>
- **commit：** `fa13908`（2026-09-21，"Initial AMC training and spherical inference release"）
- **默认分支：** `main`
- **可见性：** public
- **许可证：** 仓库根目录未放置独立 LICENSE 文件；README 声明「code is provided for research use」，上游依赖 OpenPI 各自许可
- **项目页：** <https://developer.deeprobotics.cn/research/articles/AMC_web_handoff_20261012/index.html>（及 GitHub Pages <https://zjq19990202-spec.github.io/AMC/>）
- **论文：** <https://arxiv.org/abs/2609.15012>
- **论文归档：** [amc_arxiv_2609_15012.md](../papers/amc_arxiv_2609_15012.md)
- **项目页归档：** [deeprobotics-amc.md](../sites/deeprobotics-amc.md)
- **Wiki 主实体：** [AMC（arXiv:2609.15012）](../../wiki/entities/paper-amc.md)
- **核查日期：** 2026-11-21
- **核对范围：** 拉取 `zjq19990202-spec/AMC` @ `fa13908`，通读 `README.md`、`pyproject.toml`、`scripts/train_atomic_zt_fast.py`、`scripts/train_atomic_pi05.py`、`scripts/train_force_encoder_b1.py`、`scripts/train_force_stage_b2.py`、`scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py` 与 `scripts/serve_force_pi05_keyboard_2x25.py` 头部与核心推理类 `ForcePi05Policy5x10`，并查看 `src/atomic_latent_vla/pi05/` 模块清单。边界：未安装 OpenPI JAX 环境、无数据集与 checkpoint，未运行训练/推理。
- **通读：** 已完成

## 仓库覆盖范围

仓库基于 OpenPI JAX/Flax π₀.₅ 环境扩展，仅发布训练与球面力推理源码（不含原始示范数据集、归一化统计、checkpoint 参数、标定数据与分析脚本）：

| 阶段 / 用途 | 实际脚本入口 | 说明 |
|---|---|---|
| 核心模型与数据层 | `src/atomic_latent_vla/pi05/` | 含 `model.py`、`config.py`、`force.py`、`training_data.py`、`force_training_data.py`、`gemma_adapter.py`、`weights.py` |
| 阶段 A：无视觉原子落地 (`z_T`) | `scripts/train_atomic_zt_fast.py` | 无图像/视频输入，基于 `pi05_base`、TCP twist/joint delta 目标及可选 Dual Top-2 sidecar，热启后冻结码本 |
| 阶段 B：全观测延续训练 (`z_M`) | `scripts/train_atomic_pi05.py` | 读入冻结 `z_T` teacher sidecar，配合 3 路相机图像、16 维状态与原子条件训练全观测 `z_M` 与动作专家 |
| 阶段 C1：力编码器预训练 (B1) | `scripts/train_force_encoder_b1.py` | 训练共享慢/快 120 Hz 力-状态编码器与未来力增量解码器 |
| 阶段 C2：RTC 力适应 (B2) | `scripts/train_force_stage_b2.py` | 加载 B1 checkpoint，在偏移 `{0, 10, 20, 30, 40}` 下训练动作专家与力适配器 |
| 球面力推理服务 (5×10 RTC) | `scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py` | 转发至 `serve_force_pi05_keyboard_2x25.py`（`--model-recipe spherical-b2-final`），以 `ForcePi05Policy5x10` 暴露 3 路相机 + 120 Hz 力历史 WebSocket 服务，分 `slow_prepare` 与 `fast_update` 两阶段推理 |
| 无力层级键盘服务 | `scripts/serve_layerwise_pi05_keyboard.py` | 无力自适应的层级子任务提示 WebSocket 服务入口 |

## 开源与复现边界

- **训练与服务源码完整可检视（`fa13908`）：** 四段训练与 WebSocket 服务入口清晰且均使用 OpenPI JAX/Flax 原生分片与 `nnx_utils.module_jit`。
- **数据、权重与机械臂桥接需自备：** 仓库刻意不分发 CR1 录制数据、`pi05_base` 权重、`norm_assets`、`force_norm.json`、子任务 sidecar 及硬件端 120 Hz 力传感器/相机 WebSocket 客户端桥接代码。
