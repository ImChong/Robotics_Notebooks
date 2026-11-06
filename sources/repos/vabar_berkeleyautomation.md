# BerkeleyAutomation/VABAR — 仓库状态归档

> 来源归档（repo）

- **类型：** repo
- **代码：** <https://github.com/BerkeleyAutomation/VABAR>
- **论文：** arXiv:2611.06881
- **入库日期：** 2026-11-06
- **开放状态：** **不可达**。`git ls-remote` 返回 "could not read Username"，GitHub API 返回 HTTP 403；项目页标注 available by Nov 9。许可证、目录结构、入口脚本均无法核实。

## 无法阅读 VABAR 本体，改读相关公开仓库（静态阅读、未运行）

### robosimstudio/robosimstudio（Apache-2.0，提交 `013b7181d3f5810cd29f954e00131bf6d7b97b61`，2026-11-05）

- 项目页把它列为 VABAR 的 Simulation Platform；README 称其为"重建真实机器人操作任务的开放交互式仿真工具包"，论文 arXiv:2610.18835（作者与 VABAR 重叠）。库内已有 [robosimstudio](../../wiki/entities/robosimstudio.md)，本文不新建同仓库节点。
- 与 VABAR 的可见联系：`examples/08_demos/route_the_cable/board.py` 自称复现 VABAR `cable_route/port_easy`（600 mm、50 个胶囊的线缆，两个 ManipulationNet F1 线轴，带挡板的接口）；`robosimstudio_bench/run/fleet.py` 提到 `vabar_eval.wave_widths`；`assets/embodiments/panda/README.md` 与 `tests/test_panda_vendored.py` 说明 `vabar/` 集合（含 `vab_tool_sort`）"不在本仓库"；`tests/test_demo_fold_the_towel.py` 断言公共 demo 不得引用 `vabar`。即 **VABAR 任务集合被刻意排除在该仓库之外**。
- 评测通路（可供推断 VABAR 的 Newton 实现形态，**推测**）：`scripts/bench_run.py`（`--task --solve {idle,graph,routine,vla} --episodes --workers --num-worlds --jobs`）→ `robosimstudio_bench/run/runner.py`（`RunnerEval.run`、`rollout`）/ `run/pool.py`（多进程）/ `run/fleet.py`（多世界）→ `envs/env.py`（`Env.reset/step`，`taskbuild/episodes.py:make_episode` 以 `episodes.seed + episode_id` 取种子，采样器 `fixed` / `grid_xy_quat` / `move_near` / `compose`）→ `robosimstudio_policy_agent/graph_solution.py`（`SolutionGraph`，在独立线程调用 `gap.execute`，经 `SimBus` 服务工具调用）。`taskdoc/` 为 `.task` 声明式文档（解析 / 类型检查），`goals/` 为目标树与评测器。
- 仓库自述：出货套件默认 `EpisodeSpec(count=1, sampler=fixed)`，布局变化需自带采样器的 job 文件。

### isaac-sim/IsaacLab-Arena（Apache-2.0，提交 `4f6958204266e76e7c86ab4cba1754d76d5c2a31`，状态 alpha v0.3）

- README：以 scene / embodiment / task 三元组由 `ArenaEnvBuilder` 构建 Isaac Lab 环境，支持关系型物体放置、受控环境变化（光照、背景、相机、质量等）、GPU 并行评测与敏感性分析；入口 `isaaclab_arena/evaluation/policy_runner.py`。
- `isaaclab_arena_environments/` 含 `sorting_environment.py`、`tabletop_gearmesh_environment.py`、`gear_insertion_environment.py` 等；`git ls-remote --heads` 另见 `qianl/feature/industrial-tool-sort`、`peterd/feature/v31-yam-cable-routing` 等分支，名称与 VABAR 任务相近，但**未阅读其内容、无法确认属于 VABAR**（仅为推测线索）。主分支 grep 未见 `vabar` 字样。

## 对 wiki 的映射

- [paper-vabar-variational-automation-benchmark](../../wiki/entities/paper-vabar-variational-automation-benchmark.md)
- [robosimstudio](../../wiki/entities/robosimstudio.md)
- [isaac-lab-arena](../../wiki/entities/isaac-lab-arena.md)
