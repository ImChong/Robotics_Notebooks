# OmniContact（CF-Track：Isaac Lab 训练源码）

> 官方训练仓库源码核对归档

- **标题：** OmniContact CF-Track training
- **类型：** repo
- **URL：** <https://github.com/Ingrid789/OmniContact>
- **论文：** <https://arxiv.org/abs/2606.26201v1>
- **项目页：** <https://omnicontact.github.io/>
- **部署仓库：** <https://github.com/Ingrid789/OmniContact_sim2sim>
- **核对日期：** 2026-11-21
- **核对范围：** 官方 README，以及 `scripts/rsl_rl/train.py`（motion_file_dir、gym.make、AMP 专家数据、runner.learn）与 `scripts/rsl_rl/play.py`（checkpoint 加载、回放与 ONNX 导出）入口；未在本环境执行 GPU 训练。
- **一句话说明：** CF-Track 训练代码已公开，基于 Isaac Sim 4.5.0、Isaac Lab 2.1.1 和随仓库提供的 RSL-RL fork，可训练 PPO 或 PPO+AMP 的 MLP/Transformer 策略。

## 开放范围与入口

| 位置 | 已核实内容 |
|---|---|
| `scripts/rsl_rl/train.py` | `AppLauncher` 启动 Isaac Sim → 读取 NPZ → `gym.make` 创建任务 → 配置 AMP（若启用）→ `runner.learn` |
| `scripts/rsl_rl/play.py` | 同配置任务回放，加载匹配 checkpoint，导出 ONNX |
| `source/omnicontact/` | Isaac Lab task 与本地扩展 |
| `rsl_rl/` | 包含 AMP 与 Transformer 适配的 RSL-RL fork |
| `assets/` | 训练需额外下载 README 指向的动作与 G1 资产 |
| `logs/rsl_rl/` | 训练日志与 checkpoint，具体以当前 README 为准 |

README 中的最小训练命令（需要完成 Isaac Sim、Isaac Lab 和资产安装）：

```bash
python scripts/rsl_rl/train.py \
  --task OmniContact-AMP-Transformer --num_envs 64 --max_iterations 10 \
  --motion_file_dir assets/data/box/case2_push/push \
  --disable_wandb --headless
```

支持 `OmniContact`、`OmniContact-AMP`、`OmniContact-Transformer`、`OmniContact-AMP-Transformer`；回放入口 `scripts/rsl_rl/play.py` 使用对应的 `-play` 任务名与 `--checkpoint`，回放会导出 `exported/policy.onnx`。官方 README 还列有 MLP / Transformer 预训练 checkpoint 的下载入口。

## 边界

- **已发布：** CF-Track 训练代码、MuJoCo CF-Gen/Sim2Sim 部署代码、预训练 checkpoint 下载入口；HF 数据集与额外资产按各自上游条件获取。
- **尚未本地验证：** GPU/Isaac Lab 环境安装、端到端训练收敛、真机运行与硬件闭环；这篇归档不承诺本地重现论文指标。
- **版本区分：** 本仓库训练入口与 sim2sim 仓库的 ONNX 文件名不同，不应把训练回放导出名 `exported/policy.onnx` 误写成 sim2sim 仓库现行默认文件名。

## 对 wiki 的映射

- [OmniContact 论文实体](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
- [sim2sim 源码归档](./omnicontact-sim2sim.md)
- [论文原文摘录](../papers/omnicontact_arxiv_2606_26201.md)
