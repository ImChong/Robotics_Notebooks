# isaac_asimov

> 来源归档

- **标题：** Isaac Asimov — Asimov 1 官方 Isaac Lab 行走训练扩展
- **类型：** repo
- **组织：** [Menlo Research](https://menlo.ai/)（GitHub `menloresearch`）
- **链接：** https://github.com/menloresearch/isaac_asimov
- **许可证：** BSD-3-Clause
- **Stars：** ~97（2026-09-30，GitHub API）
- **入库日期：** 2026-09-30
- **一句话说明：** Menlo 发布的 **Asimov 1** 双足 locomotion **官方 RL 训练框架**：独立 Isaac Lab extension，子模块拉取 **Isaac Lab** 与 **`asimov-1` 机器人模型**；默认 **RSL-RL** 训练 **PPO** 与 **AMP（对抗运动先验）** 速度跟踪，README 面向 **仿真训完部署真机**。
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-asimov.md`](../../wiki/entities/isaac-asimov.md)

---

## 开源状态（步骤 2.5）

- **已开源：** 训练 / 回放 / ONNX 导出 CLI、任务注册（`Asimov1-Velocity-*`）、AMP 算法与参考 motion（仓内 `.npz`）；`quick_install.sh` 一键装 **uv + Isaac Sim 5.1.0 + 钉版 Isaac Lab**。
- **项目页：** [menlo.ai](https://menlo.ai/) / [订购 Asimov 1](https://menlo.ai/order) — 商业硬件入口；**代码以本 GitHub 仓为准**（非仅 PDF 宣称）。
- **硬件同源：** 机器人资产子模块 `third_party/asimov-1`；全栈 CAD/MuJoCo/板载见 [`asimov-v1.md`](./asimov-v1.md)（`asimovinc/asimov-v1`）。

## 入口速查（README · 2026-09-30）

| 命令 / 任务 id | 作用 |
|----------------|------|
| `./quick_install.sh` | 新机器：uv 环境 + submodule（Isaac Lab、`asimov-1`）+ 依赖 |
| `./isaac_asimov.sh --train --task Asimov1-Velocity-AMP-v0 --num_envs 4096 --headless` | **AMP 推荐** baseline 单卡训练 |
| `./isaac_asimov.sh --train --task Asimov1-Velocity-v0 …` | 纯 **PPO** baseline |
| `./isaac_asimov.sh --play --task Asimov1-Velocity-AMP-Play-v0 --num_envs 32` | 加载 checkpoint 可视化；`--onnx-output` 额外导出 ONNX |
| `python -m torch.distributed.run … scripts/rsl_rl/train.py --distributed` | 多卡分布式 |
| `./isaac_asimov.sh --train … --max_iterations 100 --num_envs 128` | ~10 min 冒烟（README 称 4090 量级） |

**日志路径：** `logs/rsl_rl/<experiment_name>/<run>/`

## 依赖与版本钉（INSTALL.md）

| 组件 | 公开钉版本 / 说明 |
|------|-------------------|
| Isaac Sim | PyPI `isaacsim[all,extscache]==5.1.0` |
| PyTorch | `torch==2.7.0` + cu128（uv 路径） |
| Isaac Lab | submodule `third_party/IsaacLab`，期望 **main（文档写 2.3.2 + RSL-RL 5 支持）** |
| RSL-RL | **5.0.1**（与 Lab 安装脚本一致） |
| OS | Ubuntu 22.04+ x86_64；需 NVIDIA 驱动 |

## 代码结构（便于复现定位）

| 路径 | 含义 |
|------|------|
| `source/isaac_asimov/isaac_asimov/tasks/locomotion/` | Velocity / AMP 环境 cfg、MDP 奖励与观测 |
| `source/isaac_asimov/isaac_asimov/algorithms/amp_ppo.py` | AMP + PPO 训练逻辑 |
| `source/isaac_asimov/isaac_asimov/assets/robots/asimov_1.py` | Asimov 1 资产定义 |
| `scripts/rsl_rl/train.py` | RSL-RL 训练入口（与 `./isaac_asimov.sh` 包装） |
| `third_party/asimov-1` | 机器人模型子模块 |

## 与相邻仓库对照

| 维度 | **isaac_asimov**（本仓） | [asimov-mjlab](./asimov-mjlab.md) | [asimov-v1](./asimov-v1.md) |
|------|-------------------------|-----------------------------------|----------------------------|
| 维护方 | Menlo Research | Asimov Inc.（`asimovinc`） | Asimov Inc. 硬件主仓 |
| 仿真栈 | **Isaac Sim + Isaac Lab** | **MuJoCo Warp / mjlab** | MuJoCo 主仓 + 板载 |
| 算法亮点 | **AMP + PPO**（官方 baseline） | PPO + **1.25 Hz gait imitation** shaping | 无内置 RL 脚本 |
| 典型 env 规模 | 4096（A6000 / RTX PRO 6000 README） | 4096（mjlab 并行） | 单机 MuJoCo |

## 对 wiki 的映射

- [isaac-asimov.md](../../wiki/entities/isaac-asimov.md)
- 硬件与 MuJoCo 线：[asimov-v1.md](../../wiki/entities/asimov-v1.md)
- Isaac Lab 底座：[isaac-lab.md](../../wiki/entities/isaac-lab.md)
- mjlab 并行训练线：[asimov-mjlab.md](./asimov-mjlab.md) → 已并入 asimov-v1 叙述

## 官方延伸资源

- [README（main）](https://github.com/menloresearch/isaac_asimov/blob/main/README.md)
- [INSTALL.md（main）](https://github.com/menloresearch/isaac_asimov/blob/main/INSTALL.md)
- Menlo Discord（社区试策略上真机）：README 链接 `discord.gg/3wTVbHabtn`
