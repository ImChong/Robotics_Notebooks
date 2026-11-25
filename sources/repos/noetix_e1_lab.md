# noetix_e1_lab

> 来源归档（国内具身开源全景）

- **标题：** noetix_e1_lab
- **类型：** repo
- **机构：** 松延动力
- **链接：** https://github.com/Noetix-Robotics/noetix_e1_lab
- **分类：** 运动RL/技能训练
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `Noetix-Robotics/noetix_e1_lab@5a1d62d`（核对 `README.md`、`scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`、`scripts/sim2sim_12dof.py`、`scripts/sim2sim_mimic.py` 入口及依赖；未在本地启动 Isaac Sim 5.0 / GPU 训练）。
- **通读：** 已完成
- **一句话说明：** 松延动力基于 Isaac Lab + RSL-RL 开源的 E1 人形机器人 RL 运动与动作模仿训练框架（BSD-3-Clause），包含 `WalkRun12dof-E1-v0`（AMP-PPO）与 `Mimic-E1-v0` 任务及 MuJoCo Sim2Sim 验证脚本。
- **沉淀到 wiki：** [`wiki/entities/cn-os-noetix-e1-lab.md`](../../wiki/entities/cn-os-noetix-e1-lab.md)

## 开源状态

- **已开源**：公开仓库（BSD-3-Clause License，<https://github.com/Noetix-Robotics/noetix_e1_lab>）。

## 一手仓库要点摘录（`Noetix-Robotics/noetix_e1_lab@5a1d62d`）

- **环境栈与许可**：Isaac Sim 5.0.0 + Isaac Lab 2.2.1/2.3.0 + `rsl_rl==3.0.1` + Python 3.11；BSD-3-Clause 开源协议。
- **核心任务与管线**：先用 `play_animation.py` 预览动作重定向数据集，再通过 `scripts/rsl_rl/train.py` 训练 `WalkRun12dof-E1-v0`（AMP-PPO 风格奖励）或 `Mimic-E1-v0`（全身/动作模仿），由 `play.py` 导出 ONNX 策略，最后用 `scripts/sim2sim_12dof.py` / `scripts/sim2sim_mimic.py` 做 MuJoCo 验证。

## 对 wiki 的映射

- [wiki/entities/cn-os-noetix-e1-lab.md](../../wiki/entities/cn-os-noetix-e1-lab.md)
