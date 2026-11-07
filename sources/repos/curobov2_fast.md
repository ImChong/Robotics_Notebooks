# cuRoboV2-Fast 官方代码归档（NVlabs/curobo）

> 来源归档（repo）；官方 v0.9.0 及当前 main 的公开 Python/CUDA 代码，仅做静态分析，未实际运行 GPU benchmark。

- **标题：** cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning
- **类型：** repo
- **仓库：** <https://github.com/NVlabs/curobo>
- **官方文档：** <https://nvlabs.github.io/curobo/latest/>
- **v0.9.0 更新：** <https://github.com/NVlabs/curobo/blob/main/CHANGELOG.md>
- **论文：** <https://arxiv.org/abs/2611.06913>
- **项目：** <https://research.nvidia.com/labs/lpr/curobov2-fast/>
- **许可：** Apache-2.0（代码；附带机器人资源遵循各自许可证）
- **开源状态：** 已开源；NVIDIA 官方仓库及入门教程提供公开实现和运行入口
- **核对范围：** 仓库 README、CHANGELOG v0.9.0、官方 Motion Planning 教程、Python 入口与 MotionPlanner/TrajOpt/TimeOpt 相关实现、CUDA timeopt 目录；仅静态源码核验，未运行测试或真机。
- **通读：** 已完成
- **复核日期：** 2026-11-08
- **沉淀到 wiki：** [主节点](../../wiki/entities/paper-curobov2-fast.md)

## 官方可执行入口

```bash
# 参考 NVIDIA 官方文档（环境与依赖先按官方安装指南配置）
python -m curobo.examples.getting_started.motion_planning --mode pose
python -m curobo.examples.getting_started.motion_planning --mode grasp
```

- 入口源码：[curobo/examples/getting_started/motion_planning.py](https://github.com/NVlabs/curobo/blob/main/curobo/examples/getting_started/motion_planning.py) 的 pose/grasp 模式构建 `MotionPlannerCfg` 与 `MotionPlanner`，先 `warmup(...)`，后调用 `plan_pose(...)` 或 `plan_grasp(...)`。
- 对外 API：[curobo/motion_planner.py](https://github.com/NVlabs/curobo/blob/main/curobo/motion_planner.py) 导出 `MotionPlanner`、`MotionPlannerCfg`；[内部实现](https://github.com/NVlabs/curobo/blob/main/curobo/_src/motion/motion_planner.py) 中 `plan_pose` → `_plan_pose_single` → IK → 直线 seed 或按需 BiRRT → `TrajOptSolver.solve_pose`；单目标与 goalset 分支不同，不应画成所有请求必走 RRT。
- 时间优化：[solver_timeopt.py](https://github.com/NVlabs/curobo/blob/main/curobo/_src/solver/solver_timeopt.py) 的 `TimeOptSolver.timeopt` 将 B-spline 控制点/起点/终点传给 `TrajectoryTimeOpt.timeopt`，对输出按采样时间重建并验证速度、加速度、jerk、力矩和碰撞可行性；[motion_trajectory_timeopt.py](https://github.com/NVlabs/curobo/blob/main/curobo/_src/motion/motion_trajectory_timeopt.py) 内部 `_prepare` → `CudaTimeOptSolver.solve` → `_finish`，可通过 CUDA Graph 捕获执行。
- CUDA 实现：`curobo/_src/curobolib/kernels/trajectory/timeopt/`（局部支撑与原生加速）；`curobo/_src/trajectory/timeopt/timeopt_solver.py`（GPU TimeOpt 数值求解器）。

## 版本和复现边界

- NVIDIA 文档明确 v0.8.0 起 cuRoboV2 对 v1 发生重大 API 重构；旧的 v1 集成应锁定 v0.7.8，勿混用旧教程。
- 官方环境要求 Ubuntu、NVIDIA GPU 与兼容 CUDA/驱动；安装命令、版本约束及测试以 [Installation](https://nvlabs.github.io/curobo/latest/getting-started/installation.html) 为准。
- 本归档证明公开代码入口与对应算法模块存在，不等于已重现论文 RTX 5090/Jetson 的性能数据。论文中的真机 SceneFields/GraspGenX/YAM 系统还依赖外部组件与硬件。
