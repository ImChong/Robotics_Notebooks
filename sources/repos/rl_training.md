# rl_training（Deep Robotics）

> 来源归档

- **标题：** rl_training
- **类型：** repo
- **来源：** DeepRoboticsLab（云深处科技 Deep Robotics 官方 GitHub 组织）
- **链接：** https://github.com/DeepRoboticsLab/rl_training
- **星标（截至 2026-08-07）：** ~244
- **最近推送：** 2026-08-04
- **主要语言：** Python
- **许可证：** BSD-3-Clause
- **分类：** 强化学习训练 / Isaac Lab 厂商扩展
- **入库日期：** 2026-08-07
- **一句话说明：** 云深处官方基于 Isaac Lab 的 RL 训练扩展，注册 Lite3 / M20 / DR02 环境，默认 RSL-RL；真机部署指向同组织 `sdk_deploy` 等仓。
- **沉淀到 wiki：** 是 → [`wiki/entities/deeprobotics-rl-training.md`](../../wiki/entities/deeprobotics-rl-training.md)
- **机构：** 云深处科技（Deep Robotics）→ `deeprobotics`

---

## README 要点（编译自上游）

- 依赖徽章：**Isaac Sim 5.1.0**、**Isaac Lab 2.3.2**、**RSL-RL 5.0.1**、Python 3.11、Linux-64。
- 提供 Bilibili / YouTube 训练与部署教程视频。
- 注册环境：

| 机型 | Environment ID | 备注 |
|------|----------------|------|
| Deeprobotics Lite3 | `Rough-Deeprobotics-Lite3-v0` | 四足 rough |
| Deeprobotics M20 | `Rough-Deeprobotics-M20-v0` | 轮足 rough |
| Deeprobotics DR02 | `Amp-Flat-Deeprobotics-DR02-v0` | 平地 AMP |

- 安装：先装 Isaac Lab → 在 Lab 目录外 `git clone --recurse-submodules` → `python -m pip install -e source/rl_training` → `python scripts/tools/list_envs.py` 验证。
- 训练入口：`scripts/reinforcement_learning/rsl_rl/train.py` / `play.py`；play 支持 `--keyboard` 单机键盘遥控与 `--video` 录制。
- 仓库含 `deep_robotics_model` 子模块、`amp_locomotion_env`、`ppo_amp` 与 motion loader；另含 `LICENSE-robot_lab`，扩展形态对齐 Isaac Lab / robot_lab 一类 standalone extension。
- 部署说明：MuJoCo / 真机请用 [Deep Robotics Github Center](https://github.com/DeepRoboticsLab) 对应 deploy 仓（如 [`sdk_deploy`](https://github.com/DeepRoboticsLab/sdk_deploy)，支持 M20 / Lite3）。

## 开源状态

- **已开源**：公开 GitHub 仓库（DeepRoboticsLab/rl_training），BSD-3-Clause。
- **部署配套**：同组织 [`sdk_deploy`](https://github.com/DeepRoboticsLab/sdk_deploy) 已开源（Sim2Sim / Sim2Real，当前宣称支持 M20 与 Lite3）；本次未单独升格 wiki。

## 对 wiki 的映射

- 实体页：[`wiki/entities/deeprobotics-rl-training.md`](../../wiki/entities/deeprobotics-rl-training.md)
- 对照社区多机型扩展：[`wiki/entities/robot-lab.md`](../../wiki/entities/robot-lab.md)
- 轮足概念：[`wiki/concepts/wheel-legged-quadruped.md`](../../wiki/concepts/wheel-legged-quadruped.md)
- M20 案例：[`wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md`](../../wiki/entities/paper-aware-wheeled-legged-reflexive-evasion.md)
