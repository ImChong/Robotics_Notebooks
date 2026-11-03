# AgenticEI/VISTA

- **代码：** <https://github.com/AgenticEI/VISTA>（Apache-2.0）。
- **论文：** [arXiv:2610.35051v1](../papers/vista_arxiv_2610_35051.md)；**项目页：** <https://agenticei.github.io/VISTA-page/>（本次不可访问）。
- **开放范围：** README 称已提供 LIBERO-Pro T/S 开发运行时、仿真适配器、初始 harness、配置和离线测试；RoboCasa / RoboTwin 接入待发布；论文中的演化 harness 与实验记录不随当前源码提供。
- **运行入口：** [docs/libero.md](https://github.com/AgenticEI/VISTA/blob/main/docs/libero.md) 中 `bash tools/run-rsi.sh --config configs/local/libero.json --run runs/libero-example`。入口经 `rpent-evolve` 指向 `meta_harness.cli:main`，由 `EvolutionLoop`、`ExecutionPool` 调度开发和评估。
- **核对源码：** `pyproject.toml`、`tools/run-rsi.sh`、`meta_harness/cli.py`、`loop.py`、`meta_mcp.py`、`selection.py`，另见 [实体页](../../wiki/entities/paper-vista-active-exploration.md)。
