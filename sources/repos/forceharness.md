# ForceHarness 官方源码与运行入口

- **类型：** repo
- **代码：** https://github.com/showlab/ForceHarness
- **论文：** https://arxiv.org/abs/2611.21099
- **核对日期：** 2026-11-19
- **核对范围：** 官方 README、已公开的代码入口/脚本及许可说明；仅静态阅读，未执行训练、模型推理或真机实验。
- **通读：** 已完成
- **wiki：** [独立论文节点](../../wiki/entities/paper-forceharness-2611-21099.md)
- **项目：** https://showlab.github.io/ForceHarness/

## 开源与复现边界

`scripts/run_episode.py` 导入 `forceharness.eval.runner.run_one`、`forceharness.agent.loop.EpisodeConfig`；支持无需 API Key 的 fake backend (weight_sort scripted)，或 robosuite/付费模型实验；`configs/hiddenphys/` 为论文协议，`scripts/run_matrix.py` 跑批量。MIT。

- 官方入口：python scripts/run_episode.py --task weight_sort --agent scripted --backend fake；作者实验协议用 --paper-defaults
