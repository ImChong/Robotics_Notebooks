# What World Model Does Active Object Search Need? A Study under Imperfect Detection 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/gokyeongryeol/BAMOS
- **论文：** https://arxiv.org/abs/2611.15689
- **公开运行入口：** `uv run python scripts/isaac/run.py --methods bamos --config configs/isaac.yaml --split test`
- **相关模块：** `src/bamos 的 belief 和 planning`
- **运行产物：** episode CSV 与诊断指标
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-bamos-search-2611-15689.md)
- **原文：** [论文来源](../papers/bamos-search-2611-15689.md)

假设静态场景、近似准确相机及机器人位姿；实际移动场景与位姿漂移尚未验证。
