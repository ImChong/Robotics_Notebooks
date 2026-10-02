# Fiatlux 官方仓库

- **类型：** repo
- **代码：** <https://github.com/haw-ai-i/fiatlux>
- **项目页：** <https://fiatlux-bench.github.io/>；[核查](../sites/fiatlux.md)
- **论文：** <https://arxiv.org/abs/2609.38216>；[归档](../papers/fiatlux_arxiv_2609_38216.md)
- **遥操作数据：** [fiatlux-teleoperation](../datasets/fiatlux-teleoperation.md)
- **入库 / 核查日期：** 2026-10-02
- **沉淀到 wiki：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)

## 入口与复现顺序

| 路径 / 入口 | 作用 |
|---|---|
| `uv sync`、`assets/download_assets.sh` | 固定 Isaac Sim 5.1 / Isaac Lab 2.3.2 依赖并获取资产 |
| `scripts/list_envs.py`、`scripts/verify_scene.py` | 核查注册、传感器与场景 |
| `source/fiatlux_task/` | 环境、动作/观测、奖励、门控、灯泡 attachment 与记录 |
| `scripts/record_run.py` | 策略与环境闭环运行，保存 bag |
| `scripts/score.py`、`scripts/score_subtasks.py` | 不启动仿真的离线评分 |
| `scripts/rsl_rl/train.py`、`play.py` | PPO 训练与回放；README 明确目前仅 Replace 提供 PPO config |
| `scripts/groot/serve.sh` | 外部 GR00T PolicyServer，需单独环境与 HF 权限 |
| `scripts/teleop/sonic_teleop.py` | SONIC 下肢控制 + 双臂遥操作；额外依赖 `teleop` |

```bash
uv sync
./assets/download_assets.sh
uv run python scripts/list_envs.py
uv run python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-Replace-v0
uv run python scripts/record_run.py --task FIATLUX-S08-GrabNewBulb-v0 --policy random --episodes 20 --seed 0 --record bag --enable_cameras --out logs/runs/random0
uv run python scripts/score.py logs/runs/random0
```

以上为 README 入口核查，**本次未运行 Isaac 仿真**。相机环境需 `--enable_cameras`。环境存在和训练脚本可调用，并不证明已学会完整维护任务；真机适配仍为后续工作。
