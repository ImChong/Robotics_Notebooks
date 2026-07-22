# William-wAng618 / HumanoidArena

> 来源归档（ingest）

- **标题：** HumanoidArena — Egocentric Hierarchical Whole-body Benchmark
- **类型：** repo / benchmark
- **官方入口：** <https://github.com/William-wAng618/HumanoidArena>
- **默认分支：** `release/open-source-prep`
- **项目页：** <https://humanoidarena.github.io/>
- **论文：** <https://arxiv.org/abs/2606.17833>
- **许可：** MIT
- **入库日期：** 2026-07-22
- **一句话说明：** HumanoidArena 官方仓：TWIST2 / SONIC 遥操作与 Isaac Lab 环境、NPZ 录制与回放、LeRobot 训练集成、视觉/执行/语义 batch 评测脚本。

## 仓库布局（README）

| 路径 | 职责 |
|------|------|
| `TWIST2/` | TWIST2 控制、资产、checkpoint、机侧工具；`teleop.sh` |
| `isaaclab_twist2_g1/` | Isaac Lab 任务、replay、rerecording、`run_twist2.sh` / `run_sonic.sh`、评测入口 |
| `lerobot/` | LeRobot fork/集成（训练与 policy serving） |
| `docs/` | 环境 / 遥操作 / 数据管线 / 评测指南 |

## 关键复现路径

1. `bash isaaclab_twist2_g1/tools/setup_humanoidarena_envs.sh --dry-run` → 按 `docs/04_environment_setup.md` 安装
2. 遥操作：`cd TWIST2 && bash teleop.sh`；仿真后端 `bash isaaclab_twist2_g1/run_twist2.sh` 或 `run_sonic.sh`
3. 回放：`run_replay_twist2.sh` / `run_replay_sonic.sh`
4. 下载 ModelScope/HF 数据集与 models；按文档跑 Vision / Execution / Semantic 评测

## 对 wiki 的映射

- [HumanoidArena 实体](../../wiki/entities/paper-humanoidarena.md)
- [项目页](../sites/humanoidarena-github-io.md)
- [TWIST2](../../wiki/entities/paper-twist2.md) · [SONIC](../../wiki/methods/sonic-motion-tracking.md)
