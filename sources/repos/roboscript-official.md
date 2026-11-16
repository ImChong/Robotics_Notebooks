# RoboScript / WICL 官方实现

- **仓库：** <https://github.com/ArcheOne-AI/RoboScript>
- **论文：** <https://arxiv.org/abs/2611.12885>
- **项目：** <https://ArcheOne-AI.github.io/RoboScript/>
- **许可：** Apache-2.0（第三方 simulator 另论）
- **核验日期：** 2026-11-11
- **核对范围：** README；examples/tutorials/quickstart_mock/run.py；examples/simBenchmarks/RoboTwin/README.md 和 scripts/run_pipeline.py；内部 robotwin_pipeline 包的具体实现仍需深读。
- **通读：** 未完成

## 已核对可执行路径

1. `cd examples/tutorials && uv run python -m quickstart_mock.run`。实际 `run.py` 建立 `MockEnv`、`Hub`、`AgentSession` 和 `ScriptedBrain`，通过工具调用，最后 `env.judge`。
2. RoboTwin 实验：`examples/simBenchmarks/RoboTwin/scripts/run_pipeline.py` 组装 config 与 Wiring，调用 `robotwin_pipeline.run_pipeline`；调度 S1 技能/模仿、S2 模拟学习、P3 官方评测，GPU worker 并行。
3. 需准备独立 agent / simulator 环境、gateway token、RoboTwin 外部数据与视觉/运动服务。
4. README 中真机部署代码仍标为 TODO；不要把真机视频映射成此仓已有的可运行部署入口。

## 对 wiki 的映射

- [WICL 主节点](../../wiki/entities/paper-world-in-context-learning-roboscript.md)
