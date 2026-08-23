# CMoE（Hoshi-No-Ai / Fudan-MAGIC-Lab 官方实现）

- **标题：** CMoE — Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots
- **类型：** repo
- **仓库：** <https://github.com/Hoshi-No-Ai/CMoE>
- **镜像（README 提及）：** <https://github.com/Fudan-MAGIC-Lab/CMoE>
- **项目页：** <https://hoshi-no-ai.github.io/CMoE/>
- **论文：** arXiv:[2603.03067](https://arxiv.org/abs/2603.03067) / ICRA 2026
- **机构：** 复旦大学（Fudan University）
- **硬件：** Unitree G1（仿真 + 真机）
- **收录日期：** 2026-08-23
- **许可：** [BSD-3-Clause](https://github.com/Hoshi-No-Ai/CMoE/blob/main/LICENSE)

## 一句话摘要

复旦大学开源的 **CMoE 单阶段训练栈**：Isaac Gym Preview 4 + 定制 `rsl_rl`（`cmoe_ppo`、对比损失、双 estimator）+ `legged_gym` task **`g1cmoe`**；`train.py` / `play.py` 入口，`--alg=cmoe` 启用 MoE actor-critic 与地形对比学习。

## 为何值得保留

- **论文模块一一对应：** `cmoe_actor_critic`、`state_estimator`、`terrain_estimator`、`cmoe_on_policy_runner` 与 arXiv §III 命名对齐。
- **与上游 rsl_rl/legged_gym 不互换：** README 明确要求使用仓库内 fork，避免漏掉对比损失与 MoE 结构。
- **复现锚点：** 4096 env、5 experts、20k iter、高程图 0.7×1.1 m 等超参与论文 §IV-A 一致。

## 环境与依赖

| 组件 | 版本 / 说明 |
|------|-------------|
| OS | Ubuntu 20.04（README 测试环境） |
| Python | 3.8（conda `cmoe`） |
| PyTorch | 1.13.1 + CUDA 11.7 |
| 仿真 | NVIDIA Isaac Gym Preview 4 |
| GPU | RTX 4090，driver 550（README） |

## 目录结构（编译自 README）

```
CMoE/
├── legged_gym/        # 环境、地形、train/play
│   └── legged_gym/
│       ├── envs/      # Humanoid + G1 CMoE config（task g1cmoe）
│       ├── scripts/   # train.py, play.py
│       └── utils/
└── rsl_rl/
    └── rsl_rl/
        ├── modules/   # cmoe_actor_critic, expert_actor_critic, estimators
        ├── algorithms/# cmoe_ppo
        └── runners/   # cmoe_on_policy_runner
```

## 训练与可视化

```bash
# 安装：conda + Isaac Gym + pip install -e rsl_rl + legged_gym

python legged_gym/legged_gym/scripts/train.py --task=g1cmoe --alg=cmoe --run_name <name>
tensorboard --logdir legged_gym/logs

python legged_gym/legged_gym/scripts/play.py --task=g1cmoe --alg=cmoe
```

- `--task=g1cmoe`：Unitree G1 + CMoE 环境配置。
- `--alg=cmoe`：PPO + estimator 更新 + SwAV 式对比损失。
- checkpoint 默认写入 `legged_gym/logs/`（README 未提供公开权重 URL）。

## 真机部署（论文 §V-E，仓库未单列脚本）

- 雷达点云 + 定位 → **高程图**观测，打包送入策略网络。
- 仿真侧对高程图做延迟/高斯/椒盐噪声与倒角域随机化（式 9）；真机需自建感知对齐。

## 关联 wiki

- 实体页：[`wiki/entities/paper-cmoe.md`](../../wiki/entities/paper-cmoe.md)
- 论文归档：[`sources/papers/cmoe_contrastive_mixture_of_experts_icra_2026.md`](../papers/cmoe_contrastive_mixture_of_experts_icra_2026.md)
- 项目页：[`sources/sites/cmoe-github-io.md`](../sites/cmoe-github-io.md)
