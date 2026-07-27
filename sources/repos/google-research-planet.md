# google-research/planet

> 来源归档

- **标题：** Deep Planning Network（PlaNet）官方实现
- **类型：** repo
- **组织：** Google Research
- **代码：** <https://github.com/google-research/planet>
- **项目页：** <https://planetrl.github.io/>
- **论文：** <https://arxiv.org/abs/1811.04551>
- **License：** Apache-2.0
- **状态：** **archived**（截至 2026-07-27）
- **入库日期：** 2026-07-27
- **一句话说明：** PlaNet 开源实现：从像素学 RSSM，潜空间 CEM 规划；训练入口 `python3 -m planet.scripts.train`；依赖 TensorFlow 1.x 代栈，适合历史复现与算法对照，不宜作为新项目默认脚手架。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `python3 -m planet.scripts.train --logdir DIR --params '{tasks: [cheetah_run]}'` | 训练 |
| `scripts/tasks.py` | 任务列表 |
| `scripts/configs.py` | 默认超参 / 消融开关（`mean_only`、`model: ssm`、`planner_iterations` 等） |

## 开源状态（仓库核查，2026-07-27）

| 资产 | 状态 |
|------|------|
| 训练 / 规划代码 | **已开源** · Apache-2.0 |
| 仓库维护 | **archived** — 依赖过时风险高 |
| Disclaimer | README：非 official Google product |

## 对 wiki 的映射

- 论文：[`sources/papers/planet_latent_dynamics_arxiv_1811_04551.md`](../papers/planet_latent_dynamics_arxiv_1811_04551.md)
- 项目页：[`sources/sites/planetrl-github-io.md`](../sites/planetrl-github-io.md)
- 沉淀 **[`wiki/entities/paper-planet-latent-dynamics.md`](../../wiki/entities/paper-planet-latent-dynamics.md)**
