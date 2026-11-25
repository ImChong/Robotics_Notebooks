# Benchmarking Physical Agency (WorldLeague)｜源码归档

- **仓库：** <https://github.com/xyangzhou/WorldLeague>
- **论文：** [原文归档](../papers/worldleague.md)
- **项目页：** [官方项目](../sites/worldleague.md)
- **许可：** README 标示 MIT；资产另遵循上游条款。
- **核对日期：** 2026-11-25
- **核对范围：** README 与下述入口源码；未执行实验，完整调用链待补。
- **通读：** 未完成

## 运行入口

`worldleague` 指向 `worldleague.cli:main`；已读 `src/worldleague/cli.py`：eval 可进入 `evaluation.r1_session.evaluate`，continuous 生成 plan.json 后调用 `runtime.commands.run_continuous_episode.main`；profile 从 result.json 聚合；未执行。
