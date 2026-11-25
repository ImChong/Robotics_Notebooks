# Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control｜源码归档

- **仓库：** <https://github.com/AIGeeksGroup/MotionMamba2.0>
- **论文：** [原文归档](../papers/motion-mamba-2.md)
- **项目页：** [官方项目](../sites/motion-mamba-2.md)
- **许可：** 用户提供 Apache-2.0；仓库 LICENSE 全文待复核。
- **核对日期：** 2026-11-25
- **核对范围：** README 与下述入口源码；未执行实验，完整调用链待补。
- **通读：** 未完成

## 运行入口

`python -m scripts.simulation start` → `scripts/simulation.py` → `visualization/launch.py`；已读启动器的 environment/check/prepare/plans，六服务为 text、generator、web、physics、policy、viewer。训练入口 `python -m scripts.training_motion_mamba2`，评测 `python -m scripts.evaluate`；未执行。
