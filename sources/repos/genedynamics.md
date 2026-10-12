# GenerativeDynamics（genedynamics）代码仓库

> 来源归档

- **标题：** hhhhzl/genedynamics
- **类型：** repo
- **链接：** <https://github.com/hhhhzl/genedynamics>
- **论文：** 2GO（<https://arxiv.org/abs/2610.07772>）、MDOC（<https://arxiv.org/abs/2607.12423>）、MD-COAS（<https://arxiv.org/abs/2607.14455>）；MGA 无独立论文链接，README 指向仓库内实现
- **许可：** MIT（Copyright 2026 Zhilin He）
- **入库日期：** 2026-10-12（clone 的 HEAD `5fa3e7bb`，提交日期 2026-10-09）
- **一句话说明：** 连接生成模型、流形几何与机器人学习的模块化规划/控制/学习栈（JAX 后端，Python 3.10–3.12），内含 MDOC、MD-COAS、2GO、MGA 四个作者自研方法与 MBD、MPPI、CEM、DIAL 等基线。
- **沉淀到 wiki：** 2GO 见 [`wiki/entities/paper-2go-geometry-aware-generative-locomotion-planning.md`](../../wiki/entities/paper-2go-geometry-aware-generative-locomotion-planning.md)；同仓其他论文（MDOC、MD-COAS）以各自论文编号区分，不在本次范围内。

## 目录结构（与 2GO 相关部分）

| 路径 | 说明 |
|------|------|
| `genedynamics/solvers/single/twogo/` | 2GO 求解器（`twogo.py` 中的 `TwoGOSolver` 继承 CFSMBDSolver，另有 `schedule_contract.py`、`backends/`） |
| `genedynamics/solvers/common/{manifold,transport,model_based_diffusion}` | 流形几何、传输调度、MBD 公共组件 |
| `genedynamics/envs/` | 环境与仿真适配（MuJoCo / MJX / Brax），含四足踏脚石、人形走廊 |
| `genedynamics/experiments/runner.py` | 配置驱动的实验入口，控制台命令 `genedynamics-run`（pyproject） |
| `genedynamics/deploy/` | 部署层：AR 场景、控制器、跟随器、安全、遥操作（`genedynamics-deploy`） |
| `configs/quadruped/stepping_stones_2d/main/twogo.yaml` | 四足踏脚石 2GO 配置（horizon 16，12 维动作，active_topk 8，自适应传输调度） |
| `configs/humanoid/corridor_2d/main/twogo_zone_{a,b,c,d}.yaml`、`twogo_long.yaml` | 人形走廊 Zone A–D 配置 |
| `scripts/paper/2go/` | 论文复现脚本：`run_stepping_stones_main.sh`、`run_humanoid_corridor_main.sh`、`run_humanoid_corridor_deploy.sh`、`regen_governed_all.sh`、`plot_cost_vs_step.py` |
| `scripts/tasks/robot/humanoid/` | 人形部署相关：`governor_calibrate.py`、`audit_exec_collision.py`、`run_sport_mode_zones.py` 等 |
| `examples/plan_to_goal.py`、`examples/batched_constraints.py` | 无仿真、无 checkpoint 的 CPU/GPU 快速示例 |

## 运行入口（README / docs）

- 安装：`python -m pip install -e ".[optimization]"`；GPU/Docker 路径见 `docs/recipes/gpu-planning.md`、`docker/README.md`。
- 单例：`genedynamics-run configs/humanoid/corridor_2d/main/twogo_zone_a.yaml --device cpu --seed 0 --level 0 --development-root results/_development/v1-corridor`；`--dry-run` 校验配置。
- 论文矩阵：`bash scripts/paper/2go/run_stepping_stones_main.sh` 依次跑 mbd、mppi、ebmbd、mdoc、mdcoas、twogo 六个配置。
- 人形执行与评估：`scripts/paper/2go/run_humanoid_corridor_deploy.sh`（需 MuJoCo + PyTorch，文档建议 arm64 docker 镜像），输出 `results/humanoid/corridor_2d/deploy/<mode>/...`。
- 状态注记：docs 称 V1 支持人形走廊的**规划**配置，人形行走与推线执行仍属 deferred；README 标注为 V1 release candidate。
