# y-wng/lift（LIFT · Reactive Force VLA Post-Training）

- **标题：** LIFT — Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection
- **代码：** <https://github.com/y-wng/lift>
- **项目页：** <https://lift-policy.github.io/>
- **论文：** <https://arxiv.org/abs/2607.14236>
- **类型：** research-code（OpenPI / π₀.₅ 扩展；离线训练 + online DAgger + 策略服务）
- **机构：** 上海交通大学等（见论文）
- **依赖：** Linux、Python 3.11.x、`uv`、CUDA GPU；LeRobot 格式数据
- **首次入库：** 2026-09-27

## 一句话摘要

在 **OpenPI** 上实现 **LIFT reactive force expert**（`src/openpi/models/pi0.py` 等）：提供 **离线 π₀.₅ 对齐**、**online DAgger reactive 训练脚本**、**WebSocket 策略推理**；**不包含** Flexiv 真机驱动与 TDK 采集系统。

## 仓库边界（README 2026-09-27）

| 组件 | 仓内 | 外部 |
|------|------|------|
| 离线训练 | `scripts/train.py`、task presets（如 `pi05_iPhoneSingle_book_insertion_v3_100`） | LeRobot 数据集、`OPENPI_CHECKPOINT_ROOT` |
| Online DAgger | `scripts/train_online_dagger_lerobot_reactive.sh`、ratio / no-force-history ablation | 纠错 LeRobot 根、`OPENPI_INIT_CHECKPOINT` |
| 推理服务 | `scripts/serve_policy.py`、`openpi_client` WebSocket | 机器人观测/执行环 |
| 数据转换 | `scripts/nedf2_to_lerobot_incremental_flexiv_tdk.sh` | **TDK**、**nmx_nedf_api**、云上传 |
| Residual / vision-only 基线 | `train_online_dagger_lerobot.sh`、`train_online_dagger_lerobot_residual.sh` | 同左 |

## 关键环境变量（在线环）

- `HF_LEROBOT_HOME` / `OPENPI_CHECKPOINT_ROOT` — 数据与 checkpoint 根
- `OPENPI_LOCAL_LEROBOT_DATA_ROOT` — 在线纠错 LeRobot 导出目录
- `OPENPI_INIT_CHECKPOINT` — 离线阶段 params 路径
- Online LIFT 数据需 **6D `left_wrench`** 与 **`control_flag`**（干预 chunk 默认 `-1`）

## 对 wiki 的映射

- [`wiki/entities/paper-lift-reactive-force-vla-posttrain.md`](../../wiki/entities/paper-lift-reactive-force-vla-posttrain.md)
- [`sources/papers/lift_reactive_force_vla_arxiv_2607_14236.md`](../papers/lift_reactive_force_vla_arxiv_2607_14236.md)
- [`sources/sites/lift-policy-github-io.md`](../sites/lift-policy-github-io.md)
