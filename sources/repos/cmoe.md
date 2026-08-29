# CMoE（Hoshi-No-Ai / Fudan 官方实现）

- **标题：** CMoE — Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots
- **类型：** repo
- **仓库：** <https://github.com/Hoshi-No-Ai/CMoE>
- **组织占位仓（非可用镜像）：** <https://github.com/Fudan-MAGIC-Lab/CMoE>（截至 2026-08-29 仅空 README，`size≈1`；README 里的 `git clone Fudan-MAGIC-Lab/CMoE` **不可用**）
- **项目页：** <https://hoshi-no-ai.github.io/CMoE/>
- **论文：** arXiv:[2603.03067](https://arxiv.org/abs/2603.03067) / ICRA 2026
- **机构：** 复旦大学（Fudan University）
- **硬件：** Unitree G1（仿真 + 真机）；**12-DoF 下肢**（`g1_12dof.urdf`）
- **收录日期：** 2026-08-23
- **复核日期：** 2026-08-29
- **许可：** [BSD-3-Clause](https://github.com/Hoshi-No-Ai/CMoE/blob/main/LICENSE)；bundled `legged_gym/` / `rsl_rl/` 保留上游 BSD-3-Clause（见 `NOTICE`）

## 一句话摘要

复旦大学开源的 **CMoE 单阶段训练栈**：Isaac Gym Preview 4 + 定制 `rsl_rl`（`cmoe_ppo`、对比损失、双 estimator）+ `legged_gym` task **`g1cmoe`**；`train.py` / `play.py` 入口，`--alg=cmoe` 启用 MoE actor-critic 与地形对比学习。仓库**只覆盖仿真训练**；真机高程图与部署指向社区仓。

## 开源状态（步骤 2.5，截至 2026-08-29）

| 资源 | 状态 |
|------|------|
| `Hoshi-No-Ai/CMoE` | **已开源**（可运行 `train.py` / `play.py`） |
| 预训练 checkpoint | **未发布**（仅 `legged_gym/logs/` 自训路径） |
| 真机 onboard 包 | **未随仓发布** |
| 官方推荐高程图 | [smoggy-P/elevation_mapping_humanoid](https://github.com/smoggy-P/elevation_mapping_humanoid)（单 MID-360 LiDAR） |
| 官方推荐 G1 部署 | [fan-ziqi/rl_sar](https://github.com/fan-ziqi/rl_sar)（真机部署叠在此框架上） |
| `Fudan-MAGIC-Lab/CMoE` | **空占位**，勿当镜像克隆 |
| mjlab 移植 | [senlanke/mimic](https://github.com/senlanke/mimic) 任务 `CMoE-G1`（课程移植，非官方） |

## 为何值得保留

- **论文模块一一对应：** `cmoe_actor_critic`、`state_estimator`、`terrain_estimator`、`cmoe_on_policy_runner` 与 arXiv §III 命名对齐。
- **与上游 rsl_rl/legged_gym 不互换：** README 明确要求使用仓库内 fork，避免漏掉对比损失与 MoE 结构；代码从 [HIMLoco](https://github.com/OpenRobotLab/HIMLoco) fork。
- **复现锚点：** 4096 env、5 experts、20k iter、高程图 0.7×1.1 m、`num_prototype=32` / `temperature=0.2` 与论文 §IV-A 一致。
- **部署边界写清：** 2026-08-18 提交 `docs: update deployment references` 补了高程图与 `rl_sar` 指针。

## 环境与依赖

| 组件 | 版本 / 说明 |
|------|-------------|
| OS | Ubuntu 20.04（README 测试环境） |
| Python | 3.8（conda `cmoe`） |
| PyTorch | 1.13.1 + CUDA 11.7 |
| 仿真 | NVIDIA Isaac Gym Preview 4 |
| GPU | RTX 4090，driver 550（README） |
| 本体 | `legged_gym/resources/robots/g1/g1_12dof.urdf` |

## 目录结构（编译自 README）

```
CMoE/
├── legged_gym/        # 环境、地形、train/play
│   └── legged_gym/
│       ├── envs/      # Humanoid + G1 CMoE config（task g1cmoe；文件 g1_cmoe_config.py）
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
# 必须克隆 Hoshi-No-Ai/CMoE，不要用 Fudan-MAGIC-Lab/CMoE

python legged_gym/legged_gym/scripts/train.py --task=g1cmoe --alg=cmoe --run_name <name>
tensorboard --logdir legged_gym/logs

python legged_gym/legged_gym/scripts/play.py --task=g1cmoe --alg=cmoe
```

- `--task=g1cmoe`：Unitree G1 + CMoE 环境配置。
- `--alg=cmoe`：PPO + estimator 更新 + SwAV 式对比损失。
- checkpoint 默认写入 `legged_gym/logs/`（README 未提供公开权重 URL）。

## 真机部署（README「Deployment References」+ 论文 §V-E）

仓库本身**只做仿真训练**。作者写明真机管线叠在社区项目上，需自行对齐观测、动作、关节顺序与控制频率：

| 环节 | 官方指向 | 说明 |
|------|----------|------|
| 高程图 | [elevation_mapping_humanoid](https://github.com/smoggy-P/elevation_mapping_humanoid) | 单 MID-360 LiDAR 人形高程图 |
| 仿真/真机部署 | [rl_sar](https://github.com/fan-ziqi/rl_sar) | 「Our real-robot deployment was built on top of this framework」 |
| 感知随机化 | 论文式 9 | 高程图延迟 / 高斯 / 椒盐 / 倒角 |

## 关联 wiki

- 实体页：[`wiki/entities/paper-cmoe.md`](../../wiki/entities/paper-cmoe.md)
- mjlab 移植：[`wiki/entities/smp-g1-mjlab.md`](../../wiki/entities/smp-g1-mjlab.md)
- 论文归档：[`sources/papers/cmoe_contrastive_mixture_of_experts_icra_2026.md`](../papers/cmoe_contrastive_mixture_of_experts_icra_2026.md)
- 项目页：[`sources/sites/cmoe-github-io.md`](../sites/cmoe-github-io.md)
