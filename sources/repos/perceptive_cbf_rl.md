# perceptive_cbf_rl（PAC-MAN · lzyang2000）

> 来源归档

- **标题：** PAC-MAN — Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball
- **类型：** repo（mjlab 训练 + Unitree G1 硬件部署）
- **来源：** Caltech AMBER Lab
- **链接：** <https://github.com/lzyang2000/perceptive_cbf_rl>
- **项目页：** <https://lzyang2000.github.io/perceptive_cbf_rl/>
- **论文：** <https://arxiv.org/abs/2607.28623>
- **入库日期：** 2026-08-01
- **一句话说明：** PAC-MAN 官方实现：mjlab/MuJoCo Warp 训练、any-link 躲避球 benchmark、ZED+EfficientTAM+ONNX 真机栈；MIT。
- **开源状态：** **已开源**（2026-08-01）；根目录 `LICENSE`（MIT）；含 `deploy/ckpts/dodge_link_cbf.onnx`。
- **沉淀到 wiki：** [`wiki/entities/paper-pac-man-perceptive-cbf-rl.md`](../../wiki/entities/paper-pac-man-perceptive-cbf-rl.md)

## 仓库概况（2026-08-01）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`lzyang2000/perceptive_cbf_rl`） |
| 训练入口 | `uv sync` → `uv run python scripts/train.py`（见 `train_runs/` 每实验启动器） |
| 评测 | `scripts/dodge_benchmark.py`、`scripts/play.py` |
| 硬件 | `deploy/play_real_dodge.sh` + `deploy/ckpts/dodge_link_cbf.onnx` |
| 依赖 | Linux x86_64 · NVIDIA GPU · [uv](https://docs.astral.sh/uv/) · mjlab · rsl_rl · AMP_mjlab 适配 |
| 许可 | MIT（G1 模型与重定向动作保留上游许可） |

## README 课程映射

| 路径 | 内容 |
|------|------|
| `src/tasks/amp_loco/` | Env、奖励、CBF 项、AMP runner（mjlab manager-based） |
| `src/assets/` | G1 MJCF + 重定向 AMP clip |
| `scripts/` | `train.py` / `play.py` / `dodge_benchmark.py` / 重定向工具 |
| `train_runs/` | 论文实验格子启动器 |
| `deploy/` | ZED + EfficientTAM + ONNX + Unitree DDS |
| `tests/` | CBF 项、omni throws、gimbal aim 的 pytest |

### 主要 Task id

| Task id | 感知 |
|---------|------|
| `Unitree-G1-AMP-Dodge-MimicKit-Flat` | 状态 oracle（真值球位姿） |
| `Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat` | 固定头相机 + 球-only 掩膜深度 |
| 同上 + `CAMERA_GIMBAL=1 CAMERA_PROPRIO=1` | 云台瞄准相机 |
| `Unitree-G1-AMP-Flat` | 仅本体感觉行走（部署回路模式切换） |

### 真机一键命令（README）

```bash
STATIC_MASK=1 ETAM=1 TINY=1 NET=<your-iface> zsh deploy/play_real_dodge.sh deploy/ckpts/dodge_link_cbf.onnx
```

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | [`paper-pac-man-perceptive-cbf-rl.md`](../../wiki/entities/paper-pac-man-perceptive-cbf-rl.md) |
| 项目页 | [`perceptive-cbf-rl-github-io.md`](../sites/perceptive-cbf-rl-github-io.md) |
| 论文源 | [`pac_man_perceptive_cbf_rl_arxiv_2607_28623.md`](../papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md) |
| 仿真栈 | [`mjlab.md`](../../wiki/entities/mjlab.md)、[`amp-mjlab.md`](../../wiki/entities/amp-mjlab.md) |
| 平台 | [`unitree-g1.md`](../../wiki/entities/unitree-g1.md) |
