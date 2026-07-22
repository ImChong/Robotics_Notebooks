# HDMI（LeCAR-Lab/HDMI）

> 来源归档（repo / official code）

- **标题：** HDMI — Learning Interactive Humanoid Whole-Body Control from Human Videos
- **类型：** repo
- **URL：** <https://github.com/LeCAR-Lab/HDMI>
- **项目页：** <https://hdmi-humanoid.github.io>
- **论文：** <https://arxiv.org/abs/2509.16757>
- **机构：** LeCAR Lab / CMU
- **核查日期：** 2026-07-22
- **GitHub 状态：** 公开仓库；默认分支 `main`；GitHub API 未返回标准 licenseInfo；2026-01-17 有更新
- **一句话说明：** HDMI 官方训练代码仓，基于 IsaacSim 4.5 / IsaacLab 2.2，提供 `ppo_roa_train` / `ppo_roa_finetune` teacher-student 训练、motion replay、MuJoCo 可视化和 policy export。

## 核心摘录（归纳，非全文）

### Quick Start

- Python 3.10 conda 环境；
- `pip install "isaacsim[all,extscache]==4.5.0"`；
- IsaacLab `v2.2.0`；
- `pip install -e .` 安装 HDMI。

### 代码结构

| 路径 | 作用 |
|------|------|
| `active_adaptation/envs/` | composable MDP components 与 base env |
| `active_adaptation/learning/` | PPO 实现 |
| `active_adaptation/envs/mdp/commands/hdmi/` | HDMI commands / observations / rewards |
| `active_adaptation/learning/ppo_roa.py` | residual action distillation 对应 PPO |
| `scripts/` | train / play / visualization entrypoints |
| `cfg/` | Hydra configs |
| `data/` | motion assets and samples |

### 数据与运行入口

- `motion.npz`：body `pos/quat/lin_vel/ang_vel` 与 joint `pos/vel`；
- `meta.json`：body / joint ordering；物体作为额外 body 追加到 robot body state；
- 参考可视化：`python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase +task.command.replay_motion=true`；
- teacher：`python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase`；
- student：`python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb_run_path>`；
- 导出：play script 加 `export_policy=true`；
- sim2real：README 指向 <https://github.com/EGalahad/sim2real>。

## 对 wiki 的映射

- [HDMI 实体页](../../wiki/entities/paper-hrl-stack-06-hdmi.md)
- [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

## 参考来源（原始）

- GitHub：<https://github.com/LeCAR-Lab/HDMI>
- README（2026-07-22 抓取）：<https://raw.githubusercontent.com/LeCAR-Lab/HDMI/main/README.md>
- 项目页：<https://hdmi-humanoid.github.io>
