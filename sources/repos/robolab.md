# RoboLab（NVLabs 官方仓库）

> 来源归档

- **标题：** RoboLab
- **类型：** repo / benchmark toolkit
- **组织：** NVLabs（NVIDIA）
- **代码：** <https://github.com/NVLabs/RoboLab>
- **主页：** <https://research.nvidia.com/labs/srl/projects/robolab/>
- **论文：** <https://arxiv.org/abs/2604.09860>
- **Stars：** ~496（2026-09-14）
- **License：** Apache-2.0
- **技术栈：** Python 3.11/3.12 · uv · Isaac Sim 5.0/5.1/6.0.1（6.0.1 可选） · Isaac Lab 2.2/2.3/3.0 beta
- **入库日期：** 2026-09-14
- **一句话说明：** 基于 Isaac Lab 的通用操纵策略仿真评测：RoboLab-120 任务、server-client 策略架构、多环境并行、Claude Code skills 生成场景/任务、自包含结果 Dashboard。

## 本次官方源码核对（2026-11-17）

- **源码版本：** 官方主分支，参考 2026-11-15 `v0.3.3` Changelog：<https://github.com/NVlabs/RoboLab/blob/main/CHANGELOG.md>。
- **核对范围：** README 的安装、策略评测、Dashboard、Requirements；`policies/pi0_family/run.py`、`robolab/eval/runner.py`、`robolab/eval/episode.py`；`docs/benchmark.md`、`docs/isaac_sim_6.md`、`docs/init_pose_randomization.md`、`docs/lighting.md`、CHANGELOG。
- **通读：** 已完成
- **运行边界：** `policies/pi0_family/run.py` 调用 `run_evaluation`；`runner.py` 构建 client/环境并调用 `run_episode`；`episode.py` 执行 `infer_batch` 与 `env.step`。外部推理服务单独启动，不能认为 runner 自动启动 OpenPI server。
- **新能力：** `--init-pose-noise` 控制每回合初始物体位姿扰动（支持分组/固定物体/可复现 seed）；9 套 `lighting_selection`；可选 `--extra isaac60`（Python 3.12），默认 `isaac50`；独立扩展库 RoboLab-Long 44 项不归入原生 RoboLab-120。
- **复现提示：** PyTorch/PhysX、相机与接触动力学随 Isaac Sim 版本变化；官方推荐 RTX 48GB+ VRAM，先运行 `uv run pytest tests/`，再进行策略评估。

## 仓库结构（README）

```text
assets/objects/          物体库（312+）
assets/scenes/             场景 USD
robolab/tasks/             RoboLab-120 任务定义
robolab/robots/            机器人配置
policies/                  策略客户端（pi0_family、volo 等）
skills/robolab-scenegen/   场景生成 skill
skills/robolab-taskgen/    任务生成 skill
examples/                  空跑、回放、夹爪测试
docs/                      Dashboard、调试、生态文档
```

## 运行要点（策展）

| 项 | 说明 |
|----|------|
| 安装 | `uv sync --extra isaac50` 或 `isaac51`；两栈不可共存于同一 venv |
| 快速验证 | `uv run pytest tests/`（含单 episode 端到端） |
| 策略评测 | `uv run python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --num-envs 10` |
| 架构 | 策略 server 需另行启动；`run.py → runner.run_evaluation → episode.run_episode → client.infer_batch → env.step` |
| 并行 | `--num-envs N` 向量化多 episode |
| 生态 | [RoboVoLo](https://github.com/NVlabs/RoboVoLo) 扩展长程推理任务 |

## 对 wiki 的映射

- 实体页：[RoboLab](../../wiki/entities/robolab.md)
- 项目页：[robolab-nvidia.md](../sites/robolab-nvidia.md)
- 论文：[robolab_arxiv_2604_09860.md](../papers/robolab_arxiv_2604_09860.md)
