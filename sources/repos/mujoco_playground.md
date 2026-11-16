# google-deepmind/mujoco_playground — MuJoCo Playground 1.0

> 官方源码归档；在原有 2026-06-08 的 MJX/JAX 记录上核验 1.0，**复用原 wiki 实体**。

- **名称：** MuJoCo Playground / MuJoCo Playground 1.0
- **维护方：** Google DeepMind
- **代码：** https://github.com/google-deepmind/mujoco_playground
- **项目页：** https://playground.mujoco.org/
- **文档：** https://reaf-sim.readthedocs.io/en/latest/playground/index.html
- **PyPI：** https://pypi.org/project/playground/ （2026-11-09 已有 1.0.1）
- **许可证：** Apache-2.0（软件仓库）
- **核对版本：** 官方 `main`，2026-11-10 检索时 HEAD `66640404324ddfefd0223359a7b15974c22d6d63`；软件包 `playground>=1.0`
- **核对范围：** 官方 README 全文的技术栈、环境布局、安装、CPU/GPU、SO-101 与 G1 示例、agentic、发行说明；`rl/rsl_rl/train.py`、`play.py` 的 `main` 及入口/Runner/Viewer 调用段；`tasks/manipulation/pick_cube/so101/deploy/run.py` 的 `create_env`、`load_actor`、`main` 和 `deploy/env.py` 的仿真/真机环境构建；`tasks/teleop/so101/README.md`、`tasks/whole_body/tracking/README.md`、`tasks/agentic/tabletop_v0/README.md`。
- **通读：** 已完成
- **开源状态：** 训练、回放、遥操作与部署入口均已公开；部分真机 SDK 需可选 extras，旧 JAX/MJX 任务在 `rss2025/` 目录。
- **原入库日：** 2026-06-08
- **更新：** 2026-11-10
- **沉淀到 wiki：** [MuJoCo Playground](../../wiki/entities/mujoco-playground.md)
- **项目页归档：** [sources/sites/mujoco-playground.md](../sites/mujoco-playground.md)

## 结构与实际源码入口

| 层次 | 可核对代码 / 用途 |
|---|---|
| 训练 CLI | `python -m mujoco_playground.rl.rsl_rl.train --env=SO101PickCube --logdir=...` |
| 训练 `main` | `env_registry.get` → `entry.env_builder(nworld,device)` → `RslRlVecEnvWrapper` → `OnPolicyRunnerTB` → `runner.learn(...)` |
| 回放 CLI | `python -m mujoco_playground.rl.rsl_rl.play --env=SO101PickCube --resume_path=...` |
| 回放 `main` | `env_registry.get` → 建环境/Runner → `runner.load` / `get_inference_policy` → `reaf.simulation.torch.viewer.launch` |
| 真机 / 仿真部署 | `python -m mujoco_playground.tasks.manipulation.pick_cube.so101.deploy.run --mode=sim` 或 `--mode=real` |
| 部署 `main` | `resolve_checkpoint` 先查路径 → `create_env` → `create_policy` / `load_actor` → `runloop_lib.Runloop.run` 或 Viewer |
| 任务 / 设备分层 | `reaf-sim` 将设备侧模型和任务 logic 分开，`deploy/env.py` 负责按 mode 构建模拟环境或连接真机设备 |

训练和部署的代码路径是**两套入口**，不能画成一个不存在的 `train.py` 自动调用 `deploy.run` 的过程。

## 1.0 与 RSS2025 的关系

- **1.0 主线：** PyTorch + reaf-sim + CPU MuJoCo / MuJoCo Warp，RSL-RL 的 PPO / RL 工作流。
- **历史研究版：** 原来的 JAX/MJX 环境仍在仓库 `rss2025/`；RSS 2025 报告针对旧版基准，不应将其训练性能数值直接算作 1.0 的结果。
- **未核实：** 本次没有跑训练/真机、没有实测 4096/8192 环境吞吐或 Sim2Real 成功率。

## 对 wiki 的映射

- [MuJoCo Playground](../../wiki/entities/mujoco-playground.md) — 1.0 技术栈、图示、版本边界、复现步骤
- [MuJoCo Warp](../../wiki/entities/mujoco-warp.md) — Warp GPU 物理执行后端
- [MJX](../../wiki/entities/mujoco-mjx.md) — 历史 JAX 路线及独立保留的 MJX 能力
- [mjlab_playground](mjlab_playground.md) — 社区迁移任务与其历史来源，不等于 Playground 1.0 核心代码

## 一手链接

- [官方 README](https://github.com/google-deepmind/mujoco_playground/blob/main/README.md)
- [官方训练实现](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/rl/rsl_rl/train.py)
- [官方回放实现](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/rl/rsl_rl/play.py)
- [SO-101 部署](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/tasks/manipulation/pick_cube/so101/deploy/run.py)
- [历史 JAX 目录](https://github.com/google-deepmind/mujoco_playground/tree/main/rss2025)