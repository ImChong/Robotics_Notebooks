# ClimbLab（MATLAB 腿式攀爬仿真器）

> 来源归档

- **标题：** ClimbLab - Climbing robot Matlab simulator
- **类型：** repo
- **来源：** 东北大学（Tohoku University）空间机器人实验室 Climbing Robotics Team
- **链接：** <https://github.com/Space-Robotics-Laboratory/ClimbLab>
- **论文：** <https://arxiv.org/abs/2610.09315>
- **许可：** BSD 3-Clause
- **核查日期：** 2026-10-12（`git clone --depth 1` 浏览后删除）；最后提交 2021-08-27
- **一句话说明：** 基于 SpaceDyn 的 MATLAB 浮动基座多体仿真，含爬行/对角步态、点云地形、夹持脱离与 TSM/GIA 稳定性可视化。
- **沉淀到 wiki：** [`wiki/entities/paper-climblab-matlab-legged-climbing-sim.md`](../../wiki/entities/paper-climblab-matlab-legged-climbing-sim.md)

## 目录与入口（实际核对）

- `src/main_sim.m`：主仿真脚本，顶部 `config = 'default'` 选预设；README 写的 `src/climb_main.m` 在该提交中不存在。
- `config/`：`config_simulation.m` 汇总参数；`default/` 各模块默认参数；`preset/` 含 `clawar_2021_dynamic_climbing_demo`、`example_demo_1..3`、`nonperiodic_gait_for_discrete_footholds`、`gia_static` 等；`USER/` 用户模板。
- `src/robot/`：`ini_ANYmal_B_LP.m`、`ini_ALPHRED_LP.m`、`ini_HubRobo_*` 等机器人参数。
- `src/environment/`：`ini_surface.m`、`ini_graspable_points.m` 与点云/网格地形 `.mat`。
- `src/sensing/`、`src/controller/{gait,motion,control,equilibrium}`、`src/engine/`（`upd_fwd_dynamics.m`、`upd_ground_reaction_forces_spring_damper.m` 等）、`src/visualization/`。
- `lib/spacedyn_v2r0`：随仓库提供的 SpaceDyn；`lib/equ_gia`：GIA 稳定性。
- 要求 MATLAB 2018b 及以上（README）。

## 对 wiki 的映射

- [ClimbLab 详情节点](../../wiki/entities/paper-climblab-matlab-legged-climbing-sim.md)
- [论文来源归档](../papers/climblab_arxiv_2610_09315.md)
