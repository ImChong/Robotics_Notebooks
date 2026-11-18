# RoboScript / WICL 官方实现

- **仓库：** <https://github.com/ArcheOne-AI/RoboScript>
- **论文：** <https://arxiv.org/abs/2611.12885>
- **项目：** <https://ArcheOne-AI.github.io/RoboScript/>
- **许可：** Apache-2.0（第三方 simulator 另论）
- **核验日期：** 2026-11-18（再次核验官方 README、Mock 入口与 RoboTwin pipeline CLI）
- **核对范围：** README；examples/tutorials/quickstart_mock/run.py；examples/simBenchmarks/RoboTwin/README.md 和 scripts/run_pipeline.py 的 CLI、wiring 与调度入口；内部 robotwin_pipeline 包的完整 worker/服务调用链尚未逐函数通读。
- **通读：** 未完成

## 已核对可执行路径

1. `cd examples/tutorials && uv run python -m quickstart_mock.run`。实际 `run.py` 建立 `MockEnv`、`Hub`、`AgentSession` 和 `ScriptedBrain`，通过工具调用，最后 `env.judge`。
2. RoboTwin 实验：`examples/simBenchmarks/RoboTwin/scripts/run_pipeline.py` 组装 config 与 Wiring，调用 `robotwin_pipeline.run_pipeline`；调度 S1 技能/模仿、S2 模拟学习、P3 官方评测，GPU worker 并行。
3. 需准备独立 agent / simulator 环境、gateway token、RoboTwin 外部数据与视觉/运动服务。
4. README 中真机部署代码仍标为 TODO；不要把真机视频映射成此仓已有的可运行部署入口。

## 运行入口核验补充（2026-11-18）

```bash
git clone https://github.com/ArcheOne-AI/RoboScript.git
cd RoboScript
uv sync --all-packages
cd examples/tutorials
uv run python -m quickstart_mock.run
```

- Mock 的 `main() → asyncio.run(run(directory))` 实际构造 `MockEnv` / `Hub` / `AgentSession`，`ScriptedBrain(SCRIPT).run(session)` 调用 `sensing.get_robot_state`、`hold.hold` 和 `hub.finish`；再由**环境控制面** `env.judge(opened.episode_id)` 判定成功。该官方示例脚本的 hold 动作不会举起红盒，不应将运行成功等同于任务成功；这是环境隔离与事件日志的 smoke test。
- RoboTwin `scripts/run_pipeline.py` 首次运行通过 `compose_config(args.config, args.set)` 与 `new_directory(config)` 初始化，`wiring()` 组合 `run_s2.toolsets`、`openai_client` 和 `--worker` 子进程；`anyio.run(launch)` 调用 `robotwin_pipeline.run_pipeline`。
- 该 CLI 的 `--dry-run` **只允许与 `--resume <run_dir>` 合用**；`--summary <run_dir>` 则仅重写汇总、不启动作业。评测需独立准备 `ROBOTWIN_ROOT`、`RS_ENV_TOKEN`、模型 gateway 与视觉/运动服务。
- 截至本次核验，根目录 README 仍将 **Real-robot deployment code** 列为 TODO；仓中出现 real-robot 扩展包不等于论文 PiPER 真机实验的完整运行/安全/标定流程已经开放。

## 对 wiki 的映射

- [WICL 主节点](../../wiki/entities/paper-world-in-context-learning-roboscript.md)
