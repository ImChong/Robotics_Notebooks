# RoboScope Kit 官方代码

- **类型：** repo
- **论文：** https://arxiv.org/abs/2611.03850
- **项目页：** https://mint-sjtu.github.io/RoboScope-Kit/
- **代码：** https://github.com/MINT-SJTU/RoboScope-Kit
- **核对日期：** 2026-11-07

- **许可证：** Apache-2.0
- **核对版本：** 44fa1adb934456c40df809a7baad27152182488d
- **核对范围：** README 运行入口、benchmark CLI/orchestrator/WorkerLoop、RoboTwin worker/launcher 的动作缓存、socket 请求及观测步幅路径；并非整个仓库逐行审计。

## 入口与调用链

源码根目录下 `packages/roboscope-kit/src/roboscope_kit/`：

- `runtime/benchmark/__main__.py` → `cli.main()`；`_run_evaluation` 选择 quick/fixed/autotune，`_run_with_pool` 构造 adapter 和 `ParallelBenchmarkRuntime`，调用 `run_workers`。
- `runtime/benchmark/orchestrator.py`：模型副本池、worker 启动与监控、结果校验及 `benchmark.aggregate`。
- `runtime/benchmark/worker.py`：`WorkerLoop.run` 原子 claim，执行 callback，`queue.complete` 发布结果。
- `runtime/benchmarks/robotwin/worker.py`：`run_task` 选择持久 eval session 或单进程，启动下列 launcher，解析原生回合记录。
- 仓库根 `benchmarks/robotwin/roboscope_robotwin_launcher.py`：`ModelClient.reset/step/end_episode` 通过 `PolicySocketClient.request` 访问副本；`needs_observation` 在随机光、视频、viewer、点云开启时保留逐步观测；缓存 action_chunk，每控制步取下一动作。

[固定版本入口](https://github.com/MINT-SJTU/RoboScope-Kit/blob/44fa1adb934456c40df809a7baad27152182488d/packages/roboscope-kit/src/roboscope_kit/runtime/benchmark/cli.py)；[固定版本 RoboTwin launcher](https://github.com/MINT-SJTU/RoboScope-Kit/blob/44fa1adb934456c40df809a7baad27152182488d/benchmarks/robotwin/roboscope_robotwin_launcher.py)。

## 运行边界

Python ≥3.11；Linux x86_64 与 GPU 用于真实评测，先构建各 benchmark runtime。pip 包不含所有数据资产。当前版本多于论文的六类基准；版本变化与论文性能结论分开记录。此次未运行仿真或加载模型。

- [论文归档](../papers/roboscope_kit_arxiv_2611_03850.md)
- [项目页归档](../sites/roboscope-kit.md)
- [实体与时序图](../../wiki/entities/paper-roboscope-kit.md)
