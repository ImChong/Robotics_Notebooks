# booster_mjlab

> 来源归档

- **标题：** booster_mjlab — mjlab integration for Booster K1
- **类型：** repo
- **组织：** [IntelligentRoboticsLab](https://github.com/IntelligentRoboticsLab)
- **链接：** <https://github.com/IntelligentRoboticsLab/booster_mjlab>
- **项目页：** <https://intelligentroboticslab.github.io/booster_mjlab/>
- **Stars：** ~29（2026-09-21）
- **入库日期：** 2026-09-21
- **一句话说明：** whIRLwind 维护的 **Booster K1 × mjlab** 集成：K1 机器人模型、平地/ rough 速度跟踪与 motion tracking 任务、LAFAN1 重定向 AMP 管线；`uv` 驱动训练，W&B 管理 checkpoint 与 motion artifact；浏览器 WASM demo 与真机 AMP 行走视频见项目页。

## 入口速查（README · 2026-09-21）

| 命令 / 能力 | 作用 |
|-------------|------|
| `uv run list_envs` | 同步环境并列出全部注册 task id |
| `uv run train Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 --env.scene.num-envs 4096` | AMP + 对称 DA + Muon 优化器的平地速度跟踪 |
| `uv run play … --wandb-run-path your-org/mjlab/run-id` | 从 W&B 拉最新 checkpoint 回放 |
| `--agent.dataset-root` | 自定义 HF dataset（`namespace/repo`）或本地 motion 目录 |
| `uv run train Mjlab-Tracking-Flat-Booster-K1 --registry-name …` | motion imitation；motion 以 W&B artifact 管理 |
| `uv run visualize-motions --help` | 浏览器内浏览 / 编辑 motion clip |
| 任务命名 | `Flat`/`Rough` × 可选 `-Amp-` × 可选 `-DA-` × 可选 `-Muon-` × `-Parallel`（parallel-linkage 踝） |

## 依赖与运行环境

- **GPU：** 训练需 NVIDIA GPU；macOS 仅评估。
- **包管理：** [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）。
- **底层：** [mjlab](https://github.com/mujocolab/mjlab) manager-based RL API + MuJoCo Warp。
- **日志 / 资产：** Weights & Biases（checkpoint、motion artifact）；默认 motion 集 [whirlwind-ams/lafan_locomotion_k1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1)。

## 与相邻仓库对照

| 维度 | booster_mjlab | [unitree_rl_mjlab](unitree_rl_mjlab.md) | [booster_gym](booster_gym.md) | [mjlab_playground](mjlab_playground.md) |
|------|---------------|----------------------------------------|------------------------------|----------------------------------------|
| 维护方 | whIRLwind / IRL | Unitree 官方 | Booster 官方 | mujocolab |
| 目标硬件 | **Booster K1** | Unitree 7 款 | Booster T1 等 | Go1 / Booster T1 等示例 |
| 仿真栈 | mjlab | mjlab | Isaac Gym | mjlab |
| 特色 | **AMP + LAFAN1 K1 retarget**、Muon/DA 任务变体、WASM demo | 官方 ONNX→C++ 部署 | 端到端 Booster RL | Playground 任务端口 |

## 开源状态

- **已开源：** 训练 / 回放 / motion 可视化 CLI；HF 默认 locomotion 数据集。
- **待发布：** README「Deployment — Example code for deploying trained policies on the real Booster K1 is **coming soon**」（截至 2026-09-21）；项目页已展示真机 AMP 行走，但部署脚本未入库。

## 对 wiki 的映射

- [booster-mjlab.md](../../wiki/entities/booster-mjlab.md)
- 项目页：[`sources/sites/booster_mjlab-github-io.md`](../sites/booster_mjlab-github-io.md)
- 框架：[`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)、[`wiki/entities/amp-mjlab.md`](../../wiki/entities/amp-mjlab.md)
- 任务：[`wiki/tasks/humanoid-soccer.md`](../../wiki/tasks/humanoid-soccer.md)、[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)
