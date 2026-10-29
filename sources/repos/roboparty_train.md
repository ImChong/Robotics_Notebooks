# roboparty_train

> 来源归档（国内具身开源全景 · RP1 官方仓库体系复核）

- **标题：** roboparty_train — RPO / RP1 的 Isaac Lab 训练工作区
- **类型：** repo
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/roboparty_train （旧名 [`atom01_train`](atom01_train.md)，两者 HEAD 同为 `e24472e`，2026-10-27 核对）
- **分类：** 运动RL/技能训练
- **入库日期：** 2026-09-06（占位）；2026-10-27 复核并扩写（快照 `e24472e`，2026-10-26，提交说明 `bump robolab RP1 assets`）
- **一句话说明：** 薄包装工作区：仅含 `pyproject.toml` / `uv.lock` 与两个子模块（`robolab` Isaac Lab 扩展、`rsl_rl` 5.5.1 分支），提供 RPO 与 RP1 的 Flat / Rough / AMP / BeyondMimic / Interrupt（及 RPO 独有 Parkour、Getup）训练、回放与 ONNX 导出。
- **沉淀到 wiki：** 是（[`wiki/entities/roboparty-train.md`](../../wiki/entities/roboparty-train.md)，`project_id: roboparty-train`）
- **旧来源：** [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 开源状态

- **已开源（代码）**：README 徽章标 BSD-3；**根仓库未跟踪 `LICENSE` 文件**，许可证以子模块为准——`robolab/LICENSE`（版权署 Isaac Lab Project Developers 与 RoboLab Project Developers，BSD-3 条款）与 `rsl_rl/LICENSE`（ETH Zurich / NVIDIA，BSD-3）。
- **数据**：`robolab/data/motions/` 内附动作文件（`rp1_amp` 8 个 `.pkl`、`rp1_bm` 2 个 `.npz`、`rpo_amp` 30 个、`rpo_bm` 4 个）；训练数据由 [human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md) 导出再转换。
- **RP1 范围与总览口径有出入**：[RP1 总览 README](https://github.com/Roboparty/RP1) 将本仓描述为「RPO locomotion policies」，而本仓 README 与 `robolab` 已注册 `RP1-Flat`、`RP1-Rough`、`RP1-AMP`、`RP1-AMP-Play`、`RP1-BeyondMimic`、`RP1-Interrupt`；Parkour 仅注册了 `RPO-Parkour` / `RPO-Parkour-Play`（与总览「RP1 support coming soon」一致）。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `pyproject.toml`、`uv.lock` | Python `>=3.12,<3.13`；`rsl-rl-lib==5.5.1`；覆盖 `torch==2.11.0`（cu128）、`newton[sim]==1.6.1`、`mujoco-warp~=3.12.0`；Isaac Lab 以 `../IsaacLab-3.0.0` 路径源码依赖 |
| `.gitmodules` | `robolab` → `Roboparty/robolab`（固定 `fcf925c`）；`rsl_rl` → `Roboparty/rsl_rl`（固定 `ac30c1e`） |
| `README.md` / `README_CN.md` | 安装（MJWarp 默认 / PhysX 另建环境）、训练 / 回放命令、动作转换 |
| 子模块 `robolab/robolab/assets/robots/rp1_actuators.py` | `RP1Actuator`：延迟 PD + 电机转矩-速度包络 |
| 子模块 `robolab/data/robots/roboparty/rp1` | 指向 [RP1_description](rp1_description.md) |

## 对 wiki 的映射

- [wiki/entities/roboparty-train.md](../../wiki/entities/roboparty-train.md)
- 下游：[roboparty_deploy](roboparty_deploy.md)；模型：[rp1_description](rp1_description.md)
- 总览：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[roboparty](../../wiki/entities/roboparty.md)
